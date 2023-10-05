# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import logging
import os
import random
import sys
import time

import _pickle as cPickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')

from oscar.modeling.tokenization_bert import BertTokenizer
from oscar.modeling.modeling_bert import BertConfig
from oscar.modeling.modeling_oscar import OscarForSequenceClassification
from oscar.modeling.optimization import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'oscar': (BertConfig, OscarForSequenceClassification, BertTokenizer),
}


class VQADataset(Dataset):
    """ VQA Dataset """

    def __init__(self, args, split, tokenizer):
        super(VQADataset, self).__init__()

        assert split in ['train', 'val', 'test2015']

        self.args = args
        self.split = split
        self.tokenizer = tokenizer

        # load image features
        self.img_features = _load_img_features(args, split)  # args 'train'或'val'或'test2015'
        # print(self.img_features)  # 121287或2000或81434
        '''
        {
            ...,
            9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                       [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                       [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                       ...,
                       [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                       [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                       [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),  # (20, 2054)
            ...
        }
        '''

        # load questions
        self.examples = _load_dataset(args, split)
        # print(self.examples)  # 647480或10631或447793
        '''
        [
            {
                'guid': "train-0", 
                'text_a': "How many cookies can be seen?", 
                'text_b': "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple",
                'label': [1504]
                'score': [1.0], 
                'img_key': 9, 
                'q_id': 0
            },
            ...
        ]
        '''
        self.labels = _load_labels(args)
        # print(self.labels)
        '''
        [0, 835, 2421, 1, 78, ..., 3128]
        '''
        self.label_map = {label: i for i, label in enumerate(self.labels)}  # 把标签映射成标签索引
        '''
        {0: 0, 835: 1, 2421: 2, ..., 3128: 3128}
        '''

        logger.info('%s Data Examples: %d' % (split, len(self.examples)))  # 647480或10631或447793

    def tensorize_example(self,
                          text_a,  # question: 'How many cookies can be seen?',
                          img_feat,  # img_feat: (20, 2054)
                          text_b=None,  # od_labels: 'bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple',
                          cls_token_segment_id=0,
                          pad_token_segment_id=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          cls_token_at_end=False,
                          pad_on_left=False,
                          mask_padding_with_zero=True):
        tokens_a = self.tokenizer.tokenize(text_a)  # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?']

        tokens_b = None
        if text_b:  # 'bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple'
            tokens_b = self.tokenizer.tokenize(text_b)  # ['bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple']
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            if len(tokens_a) > self.args.max_seq_length - 2:  # Account for [CLS] and [SEP] with "- 2"
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [self.tokenizer.sep_token]  # # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]']
        segment_ids = [sequence_a_segment_id] * len(tokens)  # [0, 0, 0, 0, 0, 0, 0, 0]

        if tokens_b:  # ['bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple']
            tokens += tokens_b + [self.tokenizer.sep_token]  # ['how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]', 'bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple', '[SEP]']
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if cls_token_at_end:
            tokens = tokens + [self.tokenizer.cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [self.tokenizer.cls_token] + tokens  # ['[CLS]', 'how', 'many', 'cookies', 'can', 'be', 'seen', '?', '[SEP]', 'bowl', 'bro', '##cco', '##li', 'bowl', 'bowl', 'bowl', 'spoon', 'bowl', 'cake', 'bowl', 'don', '##ut', 'cake', 'bowl', 'dining', 'table', 'apple', '[SEP]']
            segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(tokens)  # 128 - 28 = 100
        if pad_on_left:
            tokens = ([self.tokenizer.pad_token] * padding_length) + tokens
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            tokens = tokens + ([self.tokenizer.pad_token] * padding_length)  #
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 2129, 2116, 16324, 2064, 2022, 2464, 1029, 100, 4605, 22953, 21408, 3669, 4605, 4605, 4605, 15642, 4605, 9850, 4605, 2123, 4904, 9850, 4605, 7759, 2795, 6207, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0]

        assert len(input_ids) == self.args.max_seq_length  # 128
        assert len(segment_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length

        # image features
        # print(img_feat)  # (20, 2054)
        if img_feat.shape[0] > self.args.max_img_seq_length:  # 20 > 50
            img_feat = img_feat[0: self.args.max_img_seq_length]  # (50, 2054)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+50
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:  # 20 < 50
            if self.args.max_img_seq_length > 0:  # 50 > 0
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 128+20
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]  # 128+20
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (50-20, 2054)
            img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((20, 2054), (30, 2054), dim=0) -> (50, 2054)
            if self.args.max_img_seq_length > 0:  # 50 > 0
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 148+30
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]  # 148+30
        # print(len(input_ids), len(segment_ids), len(input_mask), img_feat.shape)  # 128 128 178 (50, 2054)
        return input_ids, segment_ids, input_mask, img_feat

    def __getitem__(self, index):
        entry = self.examples[index]
        # print(entry)
        '''
        {
            'text_a': "How many cookies can be seen?", 
            'text_b': "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple",
            'label': [1504]
            'score': [1.0], 
            'img_key': 9, 
            'q_id': 0
        }
        '''
        # print(self.img_features)  # 121287或2000或81434
        '''
        {
            ...,
            9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                       [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                       [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                       ...,
                       [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                       [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                       [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),
            ...
        }
        '''
        img_feat = self.img_features[entry['img_key']]  # 9
        # print(img_feat)  # (20, 2054)
        '''
        tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                ...,
                [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]])
        '''
        input_ids, segment_ids, input_mask, img_feat = self.tensorize_example(text_a=entry['text_a'], img_feat=img_feat, text_b=entry['text_b'])

        # labels
        if entry['label'] is None:  # 测试集的label=None, score=None
            label_id = [0]  # 测试集的标签索引为[0]
            score = [0]  # 测试集的该标签索引对应的score为[0]
        elif len(entry['label']) == 0:  # 无答案的训练集和验证集的label=[], 即'an'=[]
            label_id = [0]  # 无答案的训练集和验证集的标签索引为[0]
            score = [0]  # 无答案的训练集和验证集的该标签索引对应的score为[0]
        else:  # 训练集和验证集的label=[1504], score=[1.0]
            # print(self.args.label_map)  # 用于将标签转换为对于的标签索引: [1504] -> [2698]
            '''
            {
                0: 0, 
                835: 1, 
                2421: 2, 
                ..., 
                3128: 3128
            }
            '''
            label_id = [self.label_map[l] for l in entry['label']]  # 训练集和验证集的标签映射成标签索引为[1504] -> [2698]
            score = entry['score']  # 无答案的训练集和验证集的该标签索引对应的score为[1.0]
        new_scores = target_tensor(len(self.label_map), label_id, score)  # 3129 [2689] [1.0]
        # print(new_scores)  # 长度为3129的list
        '''
        [0, 0, 0, 0, 0, 0, ..., 1.0, ..., 0]
        '''
        new_scores = torch.tensor(new_scores)  # (3129,)
        # question_id
        question_id = entry['q_id']  # 0
        return (
            torch.tensor(input_ids, dtype=torch.long),  # (128,)
            torch.tensor(segment_ids, dtype=torch.long),  # (128,)
            torch.tensor(input_mask, dtype=torch.long),  # (178,)
            img_feat,
            torch.tensor(new_scores, dtype=torch.float),  # (3129,)
            torch.tensor([question_id], dtype=torch.long)  # (1,)
        )  # input_ids, token_type_ids, attention_mask, img_feat, labels, question_id

    def __len__(self):
        return len(self.examples)


def _load_img_features(args, split):
    t_start = time.time()
    feat_file_name = '{}_img_frcnn_feats.pt'.format(split)  # train_img_frcnn_feats.pt或val_img_frcnn_feats.pt或val_img_frcnn_feats.pt或test2015_img_frcnn_feats.pt
    img_features = torch.load(os.path.join(args.data_dir, feat_file_name))  # oscar/datasets/vqa/2k/train_img_frcnn_feats.pt或oscar/datasets/vqa/2k/val_img_frcnn_feats.pt或oscar/datasets/vqa/2k/test2015_img_frcnn_feats.pt
    # print(img_features)  # 121287或2000或81434
    '''
    {
        ...,
        9: tensor([[0.1758, 0.0763, 0.0000,  ..., 0.9713, 0.4302, 0.6323],
                   [1.5736, 0.1314, 0.0000,  ..., 0.9983, 0.9895, 0.6937],
                   [0.0000, 0.0000, 0.0000,  ..., 0.3196, 0.0734, 0.1285],
                   ...,
                   [0.0000, 0.0000, 0.0000,  ..., 0.6330, 0.1572, 0.0793],
                   [0.0000, 0.0000, 0.0000,  ..., 0.9983, 0.2777, 0.1766],
                   [0.0000, 0.0000, 0.0000,  ..., 0.1899, 0.0905, 0.1060]]),  # (20, 2054)
        ...
    }
    '''
    t_end = time.time()
    logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    return img_features


def _load_dataset(args, split):
    if split == 'train':
        file_name = 'train2014_qla_mrcnn.json'
    elif split == 'val':
        file_name = 'val2014_qla_mrcnn.json'
    elif split == 'test2015':
        file_name = 'test2015_qla_mrcnn.json'
    else:
        raise ValueError()
    lines = json.load(open(os.path.join(args.data_dir, file_name)))  # oscar/datasets/vqa/2k/train2014_qla_mrcnn.json或oscar/datasets/vqa/2k/val2014_qla_mrcnn.json或oscar/datasets/vqa/2k/test2015_qla_mrcnn.json
    # print(lines)  # 647480或10631或447793
    '''
    [
        {"q": "How many cookies can be seen?", "o": "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple", "an": [1504], "s": [1.0], "img_id": 9}, 
        {"q": "What color are the dishes?", "o": "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple", "an": [2492], "s": [0.9], "img_id": 9},
        ...,
        {"q": "What colors are the animal?", "o": "horse horse", "an": [3107, 301], "s": [1.0, 0.9], "img_id": 581929}
    ]
    '''
    examples = []
    for (i, line) in enumerate(lines):
        example = {
            'text_a': line['q'],  # "How many cookies can be seen?"
            'text_b': line['o'].replace(';', ' ').strip(),  # "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple"
            'label': None if split.startswith('test') else line['an'],  # [1504]或[487, 2969, 2898]或None
            'score': None if split.startswith('test') else line['s'],  # [1.0]或[0.9, 0.6, 1.0]或None
            'img_key': line['img_id'],  # 9或241或1
            'q_id': int(line['q_id']) if split.startswith('test') else 0  # 0或0或1000
        }
        examples.append(example)
    # print(examples)  # 647480->634516或10631->10402或447793->447793
    '''
    [
        {
            'guid': "train-0", 
            'text_a': "How many cookies can be seen?", 
            'text_b': "bowl broccoli bowl bowl bowl spoon bowl cake bowl donut cake bowl dining table apple",
            'label': [1504]
            'score': [1.0], 
            'img_key': 9, 
            'q_id': 0
        },
        ...
    ]
    '''
    return examples


def _load_labels(args):
    ans2label = cPickle.load(open(args.label_file, 'rb'))  # oscar/datasets/vqa/cache/trainval_ans2label.pkl
    # print(ans2label)  # 长度为3129的dict
    '''
    {
        '': 0, 
        'boats': 835, 
        'not at all': 2421, 
        'name': 1, 
        'harley davidson': 78, 
        ..., 
        'stopping': 3128
    }
    '''
    label_list = list(ans2label.values())  # 长度为3129的list
    return label_list


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def target_tensor(len, labels, scores):  # 3129 [2689] [1.0]
    """ create the target by labels and scores """
    target = [0] * len  # [0, 0, 0, ..., 0]
    for id, l in enumerate(labels):
        target[l] = scores[id]  # [0, 0, 0, ..., 1.0, ..., 0]  # 第2698个索引对应的值为1.0
    return target


def compute_score_with_logits(logits, labels):  # (32,3129) (32,3129)
    logits = torch.max(logits, dim=1)[1].data  # (32,)
    one_hots = torch.zeros(*labels.size()).cuda()  # (32,3129)
    one_hots.scatter_(dim=1, index=logits.view(-1, 1), value=1)  # (32,3129), 对应的预测标签的值为1，其他为0
    scores = (one_hots * labels)  # (32,3129)*(32,3129) -> (32,3129), 对应的预测标签的值为1*score，其他为0
    return scores


def set_seed(seed, n_gpu):  # 42 1
    random.seed(seed)  # 42
    np.random.seed(seed)  # 42
    torch.manual_seed(seed)  # 42
    if n_gpu > 0:  # 1
        torch.cuda.manual_seed_all(seed)  # 42


def train(args, train_dataset, eval_dataset, teacher_model, student_model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)  # Note that DistributedSampler samples randomly
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=args.num_workers,  # 0
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)  # 647480/32=20234
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # apex fp16 initialization
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_loss = 0.0
    train_score = 0.0
    student_model.zero_grad()

    best_score = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(student_model)
    }

    # Prepare loss function
    mse_loss_fn = nn.MSELoss()

    def soft_cross_entropy(predictions, targets):
        student_likelihood = F.log_softmax(predictions, dim=-1)
        targets_probs = F.softmax(targets, dim=-1)
        return (-targets_probs * student_likelihood).mean()

    log_json = []
    for epoch in range(int(args.num_train_epochs)):
        t_start = time.time()
        for step, batch in enumerate(train_dataloader):  # (input_ids, token_type_ids, attention_mask, img_feat, labels, question_id)
            # TinyBERT
            teacher_model.eval()
            student_model.train()

            batch = tuple(t.to(args.device) for t in batch)  # (input_ids, token_type_ids, attention_mask, img_feat, labels, question_id)
            inputs = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'img_feats': None if args.img_feature_dim == -1 else batch[3],
                'labels': batch[4],
            }

            # L_ce
            ce_loss, student_logits, student_reps, student_atts = student_model(input_ids=inputs['input_ids'],
                                                                                token_type_ids=inputs['token_type_ids'],
                                                                                attention_mask=inputs['attention_mask'],
                                                                                img_feats=inputs['img_feats'],
                                                                                labels=inputs['labels'])
            with torch.no_grad():
                teacher_logits, teacher_reps, teacher_atts = teacher_model(input_ids=inputs['input_ids'],
                                                                           token_type_ids=inputs['token_type_ids'],
                                                                           attention_mask=inputs['attention_mask'],
                                                                           img_feats=inputs['img_feats'])

            # PKD-skip
            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            # L_attn
            att_loss = 0.
            new_teacher_atts = [
                teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)
            ]
            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device), student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device), teacher_att)
                tmp_loss = mse_loss_fn(student_att, teacher_att)
                att_loss += tmp_loss
            # L_embd + L_hidn
            rep_loss = 0.
            new_teacher_reps = [
                teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)
            ]
            for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                tmp_loss = mse_loss_fn(student_rep, teacher_rep)
                rep_loss += tmp_loss

            # L_pred = α * L__kd + (1 - α) * L_ce
            kd_loss = soft_cross_entropy(student_logits / args.temperature, teacher_logits / args.temperature)
            cls_loss = (1 - args.alpha) * ce_loss + args.alpha * kd_loss

            # L_s = β * (L_embd + L_hidn + L_attn) + L_pred
            loss = args.beta * (rep_loss + att_loss) + cls_loss  # L(x;θ_s;θ_t)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 1.0

            train_loss += loss.item()
            score = torch.sum(compute_score_with_logits(student_logits, inputs['labels']), 1).sum().item() / inputs['labels'].size(0)  # (32,)
            train_score += score
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step % args.logging_steps == 0 or global_step == t_total):
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: [{}/{}][{}/{}], lr: {:.6f}, loss: {:.4f} ({:.4f}), score: {:.4f} ({:.4f})".format(
                            epoch + 1, int(args.num_train_epochs), global_step, int(t_total), optimizer.param_groups[0]["lr"], loss, train_loss / global_step, score, train_score / global_step)
                        )

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (global_step % args.save_steps == 0 or global_step == t_total):
                    # Save model checkpoint
                    step_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch + 1, global_step))  # oscar/model/vqa/student/checkpoint-1-50
                    if not os.path.exists(step_checkpoint_dir):
                        os.makedirs(step_checkpoint_dir)
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(step_checkpoint_dir)
                    tokenizer.save_pretrained(step_checkpoint_dir)
                    torch.save(args, os.path.join(step_checkpoint_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", step_checkpoint_dir)

        t_end = time.time()
        # 每个epoch结束后做一次Evaluation
        # evaluation
        eval_loss, eval_score = evaluate(args, student_model, eval_dataset)
        logger.info("Train time cost: {:.3f}, * epoch: {}, "
                    "train_loss: {:.4f}, train_score: {:.4f}, "
                    "eval_loss: {:.4f}, eval_score: {:.4f}".format(t_end - t_start, epoch + 1,
                                                                   train_loss / global_step, train_score / global_step,
                                                                   eval_loss, eval_score))
        # save checkpoint
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0 and epoch > args.save_after_epoch:
            epoch_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))  # oscar/model/vqa/student/checkpoint-1
            if not os.path.exists(epoch_checkpoint_dir):
                os.makedirs(epoch_checkpoint_dir)
            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(epoch_checkpoint_dir)
            tokenizer.save_pretrained(epoch_checkpoint_dir)
            torch.save(args, os.path.join(epoch_checkpoint_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint {0} to {1}".format(epoch + 1, epoch_checkpoint_dir))
        # record the best model
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(student_model)
        epoch_log = {
            'epoch': epoch + 1,
            'eval_score': eval_score,
            'best_score': best_score
        }
        log_json.append(epoch_log)
        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as f:  # oscar/model/vqa/student/eval_logs.json
                json.dump(log_json, f)

    # Save the final(best) model checkpoint
    if args.local_rank in [-1, 0]:
        output_dir = args.output_dir  # oscar/model/vqa/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(best_model['model'], 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
    # return global_step, train_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset,
                                 num_workers=args.num_workers,  # 0
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)  # 10631/32=333

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))  # 10631
    logger.info("  Batch size = %d", args.eval_batch_size)

    # switch to evaluate mode
    model.eval()

    eval_loss = 0.0
    eval_score = 0
    eval_steps = 0
    t_start = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):  # (input_ids, token_type_ids, attention_mask, img_feat, labels, question_id)
        batch = tuple(t.to(args.device) for t in batch)  # (input_ids, token_type_ids, attention_mask, img_feat, labels, question_id)
        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'attention_mask': batch[2],
            'img_feats': None if args.img_feature_dim == -1 else batch[3],
            'labels': batch[4]
        }
        with torch.no_grad():
            loss, logits = model(**inputs)[:2]
        eval_loss += loss.item()
        score = torch.sum(compute_score_with_logits(logits, inputs['labels']), 1).sum().item() / inputs['labels'].size(0)  # (32,)
        eval_score += score
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    eval_score = eval_score / len(eval_dataloader)
    t_end = time.time()
    logger.info("Eval time cost: {:.3f}, Eval loss: {:.4f}, Eval score: {:.4f}".format(t_end - t_start, eval_loss, eval_score))
    return eval_loss, eval_score


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(['vqa', 'nlvr', 'coco_ir']))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")

    # Dataset
    parser.add_argument('--num_workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
    # Text
    parser.add_argument("--label_file", type=str, default='oscar/datasets/vqa/cache/trainval_ans2label.pkl', help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default='oscar/datasets/vqa/cache/trainval_label2ans.pkl', help="Label to Answer Dictionary")
    # parser.add_argument("--data_label_type", default='mask', type=str, help="faster or mask")
    # parser.add_argument("--use_vg", action='store_true', help="Use VG-QA or not.")
    # parser.add_argument("--use_vg_dev", action='store_true', help="Use VG-QA as validation.")
    # Image
    # parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    # parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="Image feature dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, choices=['faster_r-cnn', 'mask_r-cnn'], help="Image feature type.")
    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    # parser.add_argument("--use_img_layernorm", action='store_true', help="use img_layernorm")
    # parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    # parser.add_argument("--code_level", default='top', type=str, help="code level: top, bottom, both")

    # Model configuration
    parser.add_argument("--loss_type", default='bce', type=str, help="kl or bce or ce")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    # parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    # parser.add_argument("--optim", default='AdamW', type=str, help="AdamW or Adamax")

    # Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    # parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    # parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    # parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    # parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")

    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    # parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # Apex混合精度加速
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    # 多GPU分布式数据并行
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # TD
    parser.add_argument("--teacher_model", default=None, type=str, required=True, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    parser.add_argument('--num_hidden_layers', default=6, type=int, help="Number of layers of the student model")
    parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--beta', default=0.01, type=float, help="Intermediate features radio.")

    args = parser.parse_args()

    # Set output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, 'log.txt')), logging.StreamHandler(sys.stdout)])  # oscar/vqa/model/teacher/log.txt

    # Set CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')  # 初始化分布式环境，主要用来帮助进程间通信
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, str(args.device), args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Set task
    args.task_name = args.task_name.lower()  # vqa
    args.num_labels = 3129
    logger.info("Task Name: {}, #Labels: {}".format(args.task_name, args.num_labels))  # vqa 3129

    # Set model
    args.model_type = args.model_type.lower()  # oscar
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, OscarForSequenceClassification, BertTokenizer)
    tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
        args.teacher_model,  # oscar/model/vqa/teacher
        do_lower_case=args.do_lower_case
    )
    teacher_config = config_class.from_pretrained(  # BertConfig
        args.teacher_model,  # oscar/model/vqa/teacher
        num_labels=args.num_labels,  # 3129
        # finetuning_task=args.task_name
    )
    student_config = config_class.from_pretrained(  # BertConfig
        args.student_model,  # oscar/pretrained_models/base-vg-labels/ep_107_1192087
        num_hidden_layers=args.num_hidden_layers,  # 6
        num_labels=args.num_labels,  # 3129
        # finetuning_task=args.task_name
    )
    # new config
    teacher_config.img_feature_dim = args.img_feature_dim  # 2054
    teacher_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    teacher_config.hidden_dropout_prob = args.drop_out  # 0.3
    teacher_config.loss_type = args.loss_type  # bce
    teacher_config.classifier = args.classifier  # linear
    teacher_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # teacher_config.use_img_layernorm = args.use_img_layernorm
    teacher_config.output_hidden_states = True
    teacher_config.output_attentions = True
    #
    student_config.img_feature_dim = args.img_feature_dim  # 2054
    student_config.img_feature_type = args.img_feature_type  # faster_r-cnn
    student_config.hidden_dropout_prob = args.drop_out  # 0.3
    student_config.loss_type = args.loss_type  # bce
    student_config.classifier = args.classifier  # linear
    student_config.cls_hidden_scale = args.cls_hidden_scale  # 3
    # student_config.use_img_layernorm = args.use_img_layernorm
    student_config.output_hidden_states = True
    student_config.output_attentions = True
    teacher_model = model_class.from_pretrained(  # OscarForSequenceClassification
        args.teacher_model,  # oscar/model/vqa/teacher
        # from_tf=bool('.ckpt' in args.teacher_model),
        config=teacher_config
    )
    student_model = model_class.from_pretrained(  # OscarForSequenceClassification
        args.student_model,  # oscar/pretrained_models/base-vg-labels/ep_107_1192087
        # from_tf=bool('.ckpt' in args.student_model),
        config=student_config
    )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # 打印模型参数量
    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {:.1f}M'.format(teacher_total_params / 1000000))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {:.1f}M'.format(student_total_params / 1000000))

    teacher_model.to(args.device)
    student_model.to(args.device)

    logger.info("Training/evaluation parameters: %s", args)

    # Training
    if args.do_train:
        train_dataset = VQADataset(args, 'train', tokenizer)
        eval_dataset = VQADataset(args, 'val', tokenizer)
        train(args, train_dataset, eval_dataset, teacher_model, student_model, tokenizer)


if __name__ == "__main__":
    main()
