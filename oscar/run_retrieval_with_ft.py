# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import logging
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm, trange

sys.path.insert(0, '.')

from oscar.modeling.tokenization_bert import BertTokenizer
from oscar.modeling.modeling_bert import BertConfig
from oscar.modeling.modeling_oscar import OscarForSequenceClassification
from oscar.modeling.optimization import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from oscar.modeling.modeling_utils import WEIGHTS_NAME

import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'oscar': (BertConfig, OscarForSequenceClassification, BertTokenizer),
}


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""

    def __init__(self, args, split, tokenizer, is_train=True):
        super(RetrievalDataset, self).__init__()

        assert split in ['train', 'minival', 'val', 'test']

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.is_train = is_train

        feature_file = os.path.join(args.data_dir, '{}_img_{}_feats.pt'.format(split, args.img_feature_type))  # oscar/datasets/coco_ir/train_img_frcnn_feats.pt或oscar/datasets/coco_ir/minival_img_frcnn_feats.pt或oscar/datasets/coco_ir/test_img_frcnn_feats.pt
        self.img_features = torch.load(feature_file)  # 113287或1000或5000或5000
        '''
        {
            ...,
            134574: tensor([[ 0.0000,  2.1463,  0.0000,  ...,  0.9983,  0.7955,  0.9215],
                            [ 0.5593,  4.6677,  0.0000,  ...,  0.9983,  0.9056,  0.4651],
                            [ 1.5542,  9.7955,  1.2252,  ...,  0.4101,  0.0633,  0.2359],
                            ...,
                            [ 0.0000,  0.0896,  0.0000,  ...,  0.1881,  0.0950,  0.1881],
                            [ 0.7405,  3.7327,  0.0000,  ...,  0.4878,  0.6200,  0.3560],
                            [ 0.8907, 12.7547,  0.4234,  ...,  0.6314,  0.1375,  0.5062]]),
            418825: tensor([[3.4156e-01, 4.0807e-01, 0.0000e+00,  ..., 9.9875e-01, 9.1667e-01, 4.7146e-01],
                            [2.4551e-02, 2.3292e-02, 0.0000e+00,  ..., 7.6137e-01, 9.9833e-01, 5.7487e-01],
                            [1.3012e-01, 9.1158e-01, 0.0000e+00,  ..., 9.6293e-01, 6.8800e-01, 7.1294e-01],
                            ...,
                            [1.2698e-02, 6.2869e-01, 3.4424e-01,  ..., 8.6048e-01, 6.4186e-02, 5.3734e-02],
                            [4.6156e-01, 1.0837e-01, 8.9283e-04,  ..., 2.3653e-01, 4.0641e-01, 2.3653e-01],
                            [2.2118e+00, 3.3961e+00, 2.7367e-02,  ..., 5.0338e-01, 8.1081e-02, 8.9018e-02]])  # (37, 2054)
        }
        '''
        caption_file = os.path.join(args.data_dir, '{}_captions.pt'.format(split))  # oscar/datasets/coco_ir train_captions.pt或minival_captions.pt或val_captions.pt或test_captions.pt -> oscar/datasets/coco_ir/train_captions.pt
        self.captions = torch.load(caption_file)  # 113287或1000或5000或5000
        '''
        {
            ...,
            134574: '[
                "A table topped with four plates filled with food.", 
                "A number of plates with food and a glass on the table", 
                "A number of plates with food, spoon and glass", 
                "A variety of food on a dining room table. ", 
                "Plates of food on a table at a restaurant ", 
                "A picture of a couple plates of someone\'s lunch sitting on a table."
            ]'，
            418825: '[
                "Fruits and vegetables lay on a counter to prepare a meal.", 
                "A bag of strawberries on a table with tomatoes.", 
                "a bunch of food is laying out on a table", 
                "A table with several varieties of fruits and vegetables as well as flowers on top of it.", 
                "This kitchen table has fruits and vegetables on it"
            ]'
        }
        '''
        self.img_keys = list(self.img_features.keys())  # 长度为113287或1000或5000或5000的list
        '''
        [57870, 384029, 222016, 520950, ..., 134574, 418825]
        '''
        if not type(self.captions[self.img_keys[0]]) == list:  # not str == list, 将'[]'转为[]
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}  # 113287或1000或5000或5000
            '''
            {
                ...,
                134574: [
                    "A table topped with four plates filled with food.", 
                    "A number of plates with food and a glass on the table", 
                    "A number of plates with food, spoon and glass", 
                    "A variety of food on a dining room table. ", 
                    "Plates of food on a table at a restaurant ", 
                    "A picture of a couple plates of someone\'s lunch sitting on a table."
                ]，
                418825: [
                    "Fruits and vegetables lay on a counter to prepare a meal.", 
                    "A bag of strawberries on a table with tomatoes.", 
                    "a bunch of food is laying out on a table", 
                    "A table with several varieties of fruits and vegetables as well as flowers on top of it.", 
                    "This kitchen table has fruits and vegetables on it"
                ]
            }
            '''
        assert len(self.img_features) == len(self.captions), "the length of image features and captions does not match!"

        if args.add_od_labels:
            label_file = os.path.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))  # oscar/datasets/coco_ir/train_vg_labels.pt或oscar/datasets/coco_ir/minival_vg_labels.pt或oscar/datasets/coco_ir/val_vg_labels.pt或oscar/datasets/coco_ir/test_vg_labels.pt
            self.labels = torch.load(label_file)  # 113287或1000或5000或5000
            '''
            {
                ...,
                394940: 'table boy wall child mouth boy plate girl door face shirt head person nose eyes hand eye eye toothbrush knife handle watch handle hair eyebrow shirt hand hair food', 
                15335: 'wall people man woman bracelet head shirt watch man man napkin glass hair shirt shirt man head head man napkin hair shirt wall seat person hand hand man wrist'
            }
            '''

        if is_train:  # for train
            self.num_captions_per_img = args.num_captions_per_img_train  # 5
        else:  # for minival和val和test
            self.num_captions_per_img = args.num_captions_per_img_val  # 20
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(os.path.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.img_features = {k: self.img_features[k] for k in self.img_keys}
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
            # 是否指定了caption_indexs
            if args.eval_caption_index_file:  # minival_caption_indexs_top20.pt
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval  # cross_image_eval=False
                caption_index_file = os.path.join(args.data_dir, args.eval_caption_index_file)  # oscar/datasets/coco_ir/minival_caption_indexs_top20.pt
                self.caption_indexs = torch.load(caption_index_file)  # 1000
                '''
                {
                    184613: '[[184613, 1], [184613, 0], [414795, 3], [184613, 4], [184613, 3], [184613, 2], [57265, 4], [24097, 0], [173574, 3], [102159, 4], [24097, 2], [419144, 0], [24097, 1], [230454, 2], [423008, 2], [423008, 3], [358149, 3], [57265, 2], [173574, 2], [342146, 3]]', 
                    ...,
                    379842: '[[379842, 3], [379842, 4], [379842, 0], [379842, 2], [437564, 1], [140167, 4], [271986, 0], [304389, 3], [7320, 0], [322816, 0], [379842, 1], [265611, 1], [271986, 4], [571196, 2], [322816, 4], [265611, 4], [265611, 2], [29913, 2], [58254, 1], [257458, 1]]'
                }
                '''
                if not type(self.caption_indexs[self.img_keys[0]]) == list:  # not str == list, 将'[]'转为[]
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}  # 1000
                    '''
                    {
                        184613: [[184613, 1], [184613, 0], [414795, 3], [184613, 4], [184613, 3], [184613, 2], [57265, 4], [24097, 0], [173574, 3], [102159, 4], [24097, 2], [419144, 0], [24097, 1], [230454, 2], [423008, 2], [423008, 3], [358149, 3], [57265, 2], [173574, 2], [342146, 3]],  # 20条
                        ...,
                        379842: [[379842, 3], [379842, 4], [379842, 0], [379842, 2], [437564, 1], [140167, 4], [271986, 0], [304389, 3], [7320, 0], [322816, 0], [379842, 1], [265611, 1], [271986, 4], [571196, 2], [322816, 4], [265611, 4], [265611, 2], [29913, 2], [58254, 1], [257458, 1]]  # 20条
                    }
                    '''
            else:
                self.has_caption_indexs = False

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:  # 若is_train=False且cross_image_eval=True -> for test
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))  # 第0~24999张图片的img_idx为0, 第25000~49999张图片的img_idx为1, ... 第5000*4999*5~5000*4999*5+24999的img_idx为4999
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))  # 第0~24999张图片的cap_idx为0~24999, 第25000~49999张图片的cap_idx为0~24999, 第5000*4999*5~5000*4999*5+24999张图片的cap_idx为0~24999
            img_idx1 = cap_idx // self.num_captions_per_img  # 0~24999 / 5 -> 0~4999
            cap_idx1 = cap_idx % self.num_captions_per_img  # 0~24999 % 5 -> 0~4: 第0~24999张图片的cap_idx1为0~4, 第20~39张图片的cap_idx为0~4, ..., 第99980~5000*20-1张图片的cap_idx为0~4
            return img_idx, [self.img_keys[img_idx1], cap_idx1]  # (img_idx, [img_key, cap_idx])
        if not self.is_train and self.has_caption_indexs:  # is_train=False且has_caption_indexs=True -> for minival
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]  # (img_idx, [img_key, cap_idx])

        # for train或val
        img_idx = index // self.num_captions_per_img  # 第0~4张图片的img_idx为0, 第5~9张图片的img_idx为1, ..., 第113280~113284张图片的img_idx为22656, ..., 第566430~566434张图片的img_idx为113286
        cap_idx = index % self.num_captions_per_img  # 第0~4张图片的cap_idx为0~4, 第5~9张图片的cap_idx为0~4, ..., 第113280~113284张图片的cap_idx为0~4, ..., 第566430~566434张图片的cap_idx为0~4
        return img_idx, [self.img_keys[img_idx], cap_idx]  # (img_idx, [img_key, cap_idx]): (0, [57870, 0]), (0, [57870, 1]), (0, [57870, 2]), (0, [57870, 3]), (0, [57870, 4]), ..., (113286, [418825, 0]), (113286, [418825, 1]), (113286, [418825, 2]), (113286, [418825, 3]), (113286, [418825, 4])

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):  # 418825
        if self.args.add_od_labels:  # True
            # if type(self.labels[img_key]) == str:
            #     od_labels = self.labels[img_key]
            # else:
            #     od_labels = ' '.join([l['class'] for l in self.labels[img_key]])
            od_labels = self.labels[img_key]
            return od_labels

    def tensorize_example(self,
                          text_a,  # caption: "Fruits and vegetables lay on a counter to prepare a meal."
                          img_feat,  # img_feat: (37, 2054)
                          text_b=None,  # od_labels: "table table counter glass fruit jar shelf lid flowers tomato bag tomato tomato wall bag container cup vase strawberries flower container cutting board container lemon strawberry tomato strawberry laptop pot table bookshelf tomatoes strawberries"
                          cls_token_segment_id=0,
                          pad_token_segment_id=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          cls_token_at_end=False,
                          pad_on_left=False,
                          mask_padding_with_zero=True):
        tokens_a = self.tokenizer.tokenize(text_a)  # ['fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.']

        tokens_b = None
        if text_b:  # "table table counter glass fruit jar shelf lid flowers tomato bag tomato tomato wall bag container cup vase strawberries flower container cutting board container lemon strawberry tomato strawberry laptop pot table bookshelf tomatoes strawberries"
            tokens_b = self.tokenizer.tokenize(text_b)  # ['table', 'table', 'counter', 'glass', 'fruit', 'jar', 'shelf', 'lid', 'flowers', 'tomato', 'bag', 'tomato', 'tomato', 'wall', 'bag', 'container', 'cup', 'vase', 'straw', '##berries', 'flower', 'container', ...]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)  # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            if len(tokens_a) > self.args.max_seq_length - 2:  # Account for [CLS] and [SEP] with "- 2"
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [self.tokenizer.sep_token]  # ['fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.', '[SEP]']
        segment_ids = [sequence_a_segment_id] * (len(tokens))  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if text_b:  # "table table counter glass fruit jar shelf lid flowers tomato bag tomato tomato wall bag container cup vase strawberries flower container cutting board container lemon strawberry tomato strawberry laptop pot table bookshelf tomatoes strawberries"
            tokens += tokens_b + [self.tokenizer.sep_token]  # ['fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.', '[SEP]', 'table', 'table', 'counter', 'glass', 'fruit', 'jar', 'shelf', 'lid', 'flowers', 'tomato', 'bag', 'tomato', 'tomato', ..., '[SEP]']
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if cls_token_at_end:
            tokens = tokens + [self.tokenizer.cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [self.tokenizer.cls_token] + tokens  # ['[CLS]', 'fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.', '[SEP]', 'table', 'table', 'counter', 'glass', 'fruit', 'jar', 'shelf', 'lid', 'flowers', 'tomato', 'bag', 'tomato', ...,  '[SEP]']
            segment_ids = [cls_token_segment_id] + segment_ids  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(tokens)  # 70 - 53 = 17
        if pad_on_left:
            tokens = ([self.tokenizer.pad_token] * padding_length) + tokens
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            tokens = tokens + ([self.tokenizer.pad_token] * padding_length)  # ['[CLS]', 'fruits', 'and', 'vegetables', 'lay', 'on', 'a', 'counter', 'to', 'prepare', 'a', 'meal', '.', '[SEP]', 'table', 'table', 'counter', 'glass', 'fruit', 'jar', 'shelf', 'lid', 'flowers', 'tomato', ..., '[PAD]']
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, , 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # [101, 10962, 1998, 11546, 3913, 2006, 1037, 4675, 2000, 7374, 1037, 7954, 1012, 102, 2795, 2795, 4675, 3221, 5909, 15723, 11142, 11876, 4870, 20856, 4524, ..., 13137, 20968, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert len(input_ids) == self.args.max_seq_length  # 70
        assert len(segment_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length

        # image features
        # print(img_feat)  # (37, 2054)
        if img_feat.shape[0] > self.args.max_img_seq_length:  # 37 > 50
            img_feat = img_feat[0: self.args.max_img_seq_length]  # (50, 2054)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 70+50
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:  # 37 < 50
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]  # 70+37
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]  # 70+37
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))  # (50-37, 2054)
            img_feat = torch.cat((img_feat, padding_matrix), 0)  # torch.cat((37,2054), (13,2054)), dim=0) -> (50, 2054)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])  # 107+13
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]  # 107+13
        # print(len(input_ids), len(segment_ids), len(input_mask), img_feat.shape)  # 70 70 120 (50, 2054)

        input_ids = torch.tensor(input_ids, dtype=torch.long)  # (70,)
        token_type_ids = torch.tensor(segment_ids, dtype=torch.long)  # (70,)
        attention_mask = torch.tensor(input_mask, dtype=torch.long)  # (120,)
        return input_ids, token_type_ids, attention_mask, img_feat

    def __getitem__(self, index):
        if self.is_train:  # for train
            img_idx, cap_idxs = self.get_image_caption_index(index)  # (img_idx, [img_key, cap_idx]): (0, [57870, 0]), (0, [57870, 1]), (0, [57870, 2]), (0, [57870, 3]), (0, [57870, 4]), ..., (113286, [418825, 0]), (113286, [418825, 1]), (113286, [418825, 2]), ..., (113286, [418825, 4])
            img_key = self.img_keys[img_idx]  # 113287 -> 418825
            img_feat = self.img_features[img_key]  # 418825 -> (37, 2054)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]  # (418825, 0) -> "Fruits and vegetables lay on a counter to prepare a meal."
            od_labels = self.get_od_labels(img_key)  # 418825 -> "table table counter glass fruit jar shelf lid flowers tomato bag tomato tomato wall bag container cup vase strawberries flower container cutting board container lemon strawberry tomato ... strawberries"
            example = self.tensorize_example(text_a=caption, img_feat=img_feat, text_b=od_labels)  # ((70,), (70,), (120,), (50,2054))

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))  # [0, 1, ..., img_idx-1, img_idx+1, ..., 113286]
            img_idx_neg = random.choice(neg_img_indexs)  # 从余下的113286张图片中选择一个作为负样本
            if random.random() <= 0.5:  # 要么选择该图片的image，要么选择该图片的caption
                # randomly select a negative caption from a different image.
                cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)  # 从0~4中选择一个cap_idx
                caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]  # 取负样本图片的一条caption和原图片的image组成负样本
                example_neg = self.tensorize_example(text_a=caption_neg, img_feat=img_feat, text_b=od_labels)
            else:
                # randomly select a negative image.
                img_feat_neg = self.img_features[self.img_keys[img_idx_neg]]  # 取负样本图片的img_feat和原图片的caption组成负样本
                od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])  # 取负样本图片的object tags和原图片的caption组成负样本
                example_neg = self.tensorize_example(text_a=caption, img_feat=img_feat_neg, text_b=od_labels_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])  # ((70,), (70,), (120,), (50,2054), 1, (70,), (70,), (120,), (50,2054), 0)
            return index, example_pair  # index, (input_ids, token_type_ids, attention_mask, img_feat, label, input_ids, token_type_ids, attention_mask, img_feat, label)
        else:  # for minival或val或test
            img_idx, cap_idxs = self.get_image_caption_index(index)  # (img_idx, [img_key, cap_idx])
            img_key = self.img_keys[img_idx]
            img_feat = self.img_features[img_key]
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(text_a=caption, img_feat=img_feat, text_b=od_labels)

            label = 1 if img_key == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])  # index, (input_ids, token_type_ids, attention_mask, img_feat, label)

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:  # 若is_train=False且cross_iamge_eval=True
            return len(self.img_keys) ** 2 * self.num_captions_per_img  # for test: 5000*5000*5=125000000
        return len(self.img_keys) * self.num_captions_per_img  # for train: 113287*5=566435或for minival: 1000*20=20000


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


def compute_ranks(dataset, prediction):
    # print(prediction)
    """
    {
        0: 0.9445,
        1: 0.6422,
        ...,
        19999: 0.3254
    }
    """
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])  # (20000,)
    similarities = np.array([prediction[i] for i in range(len(dataset))])  # (20000,)
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img  # 20
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img  # 1000*20
    labels = np.reshape(labels, [-1, num_captions_per_img])  # (1000,20)
    similarities = np.reshape(similarities, [-1, num_captions_per_img])  # (1000,20)
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]  # 对sim中每行的元素（即这20条caption的相似度）进行从大到小排序并返回排序后的原索引
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:  # 若has_caption_indexs=False
        labels = np.swapaxes(labels, 0, 1)  # (20,1000)
        similarities = np.swapaxes(similarities, 0, 1)  # (20,1000)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)  # Note that DistributedSampler samples randomly
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=args.num_workers,  # 0
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)  # 566435/32=17702
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
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
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

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
    train_acc = 0.0
    model.zero_grad()

    best_acc = 0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model)
    }

    log_json = []
    for epoch in range(int(args.num_train_epochs)):
        t_start = time.time()
        for step, (_, batch) in enumerate(train_dataloader):  # index, (input_ids, token_type_ids, attention_mask, img_feat, label, input_ids, token_type_ids, attention_mask, img_feat, label)
            model.train()

            batch = tuple(t.to(args.device) for t in batch)  # (input_ids, token_type_ids, attention_mask, img_feat, label, input_ids, token_type_ids, attention_mask, img_feat, label)
            inputs = {
                'input_ids': torch.cat((batch[0], batch[5]), dim=0),  # ((32,70), (32,70), dim=0) -> (64,70)
                'token_type_ids': torch.cat((batch[1], batch[6]), dim=0),  # ((32,70), (32,70), dim=0) -> (64,70)
                'attention_mask': torch.cat((batch[2], batch[7]), dim=0),  # ((32,120), (32,120), dim=0) -> (64,120)
                'img_feats': torch.cat((batch[3], batch[8]), dim=0),  # ((32,50,2054), (32,50,2054), dim=0) -> (64,50,2054)
                'labels': torch.cat((batch[4], batch[9]), dim=0)  # ((32,), (32,), dim=0) -> (64,)
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 1.0

            train_loss += loss.item()
            acc = (logits.argmax(dim=1) == inputs['labels'].view(-1)).sum().item() / inputs['labels'].size(0)
            train_acc += acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step % args.logging_steps == 0 or global_step == t_total):
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: [{}/{}][{}/{}], lr: {:.6f}, loss: {:.4f} ({:.4f}), acc: {:.4f} ({:.4f})".format(
                            epoch + 1, int(args.num_train_epochs), global_step, int(t_total), optimizer.param_groups[0]["lr"], loss, train_loss / global_step, acc, train_acc / global_step)
                        )

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (global_step % args.save_steps == 0 or global_step == t_total):
                    # Save model checkpoint
                    step_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch + 1, global_step))  # oscar/model/coco_ir/teacher/checkpoint-1-50
                    if not os.path.exists(step_checkpoint_dir):
                        os.makedirs(step_checkpoint_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(step_checkpoint_dir)
                    tokenizer.save_pretrained(step_checkpoint_dir)
                    torch.save(args, os.path.join(step_checkpoint_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to {}".format(step_checkpoint_dir))

        t_end = time.time()
        # 每个epoch结束后做一次Evaluation
        # evaluation
        eval_loss, eval_result = evaluate(args, model, eval_dataset)
        '''
        {
            "i2t_retrieval": {
                "R@1": xxx, 
                "R@5": xxx, 
                "R@10": xxx
            },
            "t2i_retrieval": {
                "R@1": yyy, 
                "R@5": yyy, 
                "R@10": yyy
            }
        }
        '''
        # logger.info("Train time cost: {:.3f}, * epoch: {}, "
        #             "train_loss: {:.4f}, train_acc: {:.4f}, "
        #             "eval_loss: {:.4f}, "
        #             "i2t_retrieval: R@1: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10, "
        #             "t2i_retrieval: R@1: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(t_end - t_start, epoch + 1,
        #                                                                                 train_loss / global_step, train_acc / global_step,
        #                                                                                 eval_loss,
        #                                                                                 eval_result['i2t_retrieval']['R@1'], eval_result['i2t_retrieval']['R@5'], eval_result['i2t_retrieval']['R@10'],
        #                                                                                 eval_result['t2i_retrieval']['R@1'], eval_result['i2t_retrieval']['R@5'], eval_result['t2i_retrieval']['R@10']))
        # save checkpoint
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0 and epoch > args.save_after_epoch:
            epoch_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))  # oscar/model/coco_ir/teacher/checkpoint-1
            if not os.path.exists(epoch_checkpoint_dir):
                os.makedirs(epoch_checkpoint_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(epoch_checkpoint_dir)
            tokenizer.save_pretrained(epoch_checkpoint_dir)
            torch.save(args, os.path.join(epoch_checkpoint_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to {}".format(epoch_checkpoint_dir))
        # record the best model
        rank_accs = eval_result['i2t_retrieval']
        if rank_accs['R@1'] > best_acc:
            best_acc = rank_accs['R@1']
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(model)
        epoch_log = {
            'epoch': epoch,
            'i2t_retrieval': eval_result['i2t_retrieval'],
            't2i_retrieval': eval_result['t2i_retrieval'],
            'best_R1': best_acc
        }
        log_json.append(epoch_log)
        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as f:  # oscar/model/coco_ir/teacher/eval_logs.json
                json.dump(log_json, f)

    # Save the final(best) model checkpoint
    if args.local_rank in [-1, 0]:
        output_dir = args.output_dir  # oscar/model/coco_ir/teacher
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(best_model['model'], 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
    # return global_step, train_loss / global_step


def evaluate(args, model, eval_dataset, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset,
                                 num_workers=args.num_workers,  # 0
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)  # 20000/32=625或125000000/32=3906250

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = {}".format(len(eval_dataset)))  # 20000
    logger.info("  Batch size = {}".format(args.eval_batch_size))

    # switch to evaluate mode
    model.eval()

    eval_loss = 0.0
    eval_prediction = {}
    eval_step = 0
    t_start = time.time()
    for indexs, batch in tqdm(eval_dataloader):  # index, (input_ids, token_type_ids, attention_mask, img_feat, label)
        batch = tuple(t.to(args.device) for t in batch)  # (input_ids, token_type_ids, attention_mask, img_feat, label)
        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'attention_mask': batch[2],
            'img_feats': batch[3],
            'labels': batch[4]
        }
        with torch.no_grad():
            loss, logits = model(**inputs)[:2]
        eval_loss += loss.item()
        probs = nn.Softmax(dim=1)(logits)  # (32, 2)
        preds = probs[:, 1]  # (32,), the confidence to be a matched pair
        preds = [_.to(torch.device("cpu")) for _ in preds]
        eval_prediction.update({idx.item(): res.item() for idx, res in zip(indexs, preds)})
        eval_step += 1
    # print(eval_prediction)
    '''
    {
        0: 0.9445,
        1: 0.6422,
        ...,
        19999: 0.3254
    }
    '''
    eval_loss = eval_loss / eval_step
    t_end = time.time()
    logger.info('Eval time cost: {:.3f}, Eval loss: {:.4f}'.format(t_end - t_start, eval_loss))
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, eval_prediction)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    '''
    {
        "i2t_retrieval": {
            "R@1": xxx, 
            "R@5": xxx, 
            "R@10": xxx
        },
        "t2i_retrieval": {
            "R@1": yyy, 
            "R@5": yyy, 
            "R@10": yyy
        }
    }
    '''
    return eval_loss, eval_result


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(['vqa', 'nlvr', 'coco_ir']))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")

    # Dataset
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loading workers")
    # Text
    parser.add_argument("--add_od_labels", default=True, action='store_true', help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, help="label type, support vg, gt, oid")
    # Image
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="Image feature dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str, help="Image feature type.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")
    # parser.add_argument("--use_img_layernorm", action='store_true', help="use img_layernorm")

    # Model configuration
    parser.add_argument("--loss_type", default='ce', type=str, help="Loss function types: support kl, ce")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    # parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    # parser.add_argument("--optim", default='AdamW', type=str, help="AdamW or Adamax")

    # Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=70, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each save_steps.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    # parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # Apex混合精度加速
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    # 多GPU分布式数据并行
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Training
    parser.add_argument("--eval_caption_index_file", default='', type=str, help="index of a list of (img_key, cap_idx) for each image. this is used to perform re-rank using hard negative samples. useful for validation set to monitor the performance during training.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int, help="Number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int, help="Number of captions for each testing image.")
    # Inference
    parser.add_argument("--eval_img_keys_file", default='', type=str, help="image key tsv to select a subset of images for evaluation. This is useful in 5-folds evaluation. The topn index file is not needed in this case.")
    parser.add_argument("--cross_image_eval", action='store_true', help="perform cross image inference, ie. each image with all texts from other images.")
    # parser.add_argument("--eval_model_dir", type=str, default='', help="Model directory for evaluation.")

    # FT
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of layers of the student model")

    args = parser.parse_args()

    # Set output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set logging
    # logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)
    # fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))  # oscar/coco_ir/model/teacher/log.txt
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, 'log.txt')), logging.StreamHandler(sys.stdout)])  # oscar/coco_ir/model/teacher/log.txt

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
    args.task_name = args.task_name.lower()  # coco_ir
    args.num_labels = 2
    logger.info("Task Name: {}, #Labels: {}".format(args.task_name, args.num_labels))  # coco_ir 2

    # Set model
    args.model_type = args.model_type.lower()  # oscar
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (BertConfig, OscarForSequenceClassification, BertTokenizer)
    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(  # BertTokenizer
            args.model_name_or_path,  # oscar/pretrained_models/base-vg-labels/ep_67_588997
            do_lower_case=args.do_lower_case
        )
        config = config_class.from_pretrained(  # BertConfig
            args.model_name_or_path,  # pretrained_models/base-vg-labels/ep_67_588997
            num_hidden_layers=args.num_hidden_layers,  # 6
            num_labels=args.num_labels,  # 2
            # finetuning_task=args.task_name
        )
        # new config
        config.img_feature_dim = args.img_feature_dim  # 2054
        config.img_feature_type = args.img_feature_type  # frcnn
        config.hidden_dropout_prob = args.drop_out  # 0.1
        config.loss_type = args.loss_type  # ce
        config.classifier = args.classifier  # linear
        config.cls_hidden_scale = args.cls_hidden_scale  # 3
        # config.use_img_layernorm = args.use_img_layernorm
        model = model_class.from_pretrained(  # OscarForSequenceClassification
            args.model_name_or_path,  # oscar/pretrained_models/base-vg-labels/ep_67_588997
            # from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config
        )
    elif args.do_eval or args.do_test:
        checkpoint = args.output_dir
        assert os.path.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Model Parameters: {:.1f}M'.format(total_params / 1000000))

    model.to(args.device)

    logger.info("Training/evaluation parameters: %s", args)

    # Training
    if args.do_train:
        train_dataset = RetrievalDataset(args, 'train', tokenizer, is_train=True)
        eval_dataset = RetrievalDataset(args, 'minival', tokenizer, is_train=False)
        train(args, train_dataset, eval_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        eval_dataset = RetrievalDataset(args, 'minival', tokenizer, is_train=False)
        evaluate(args, model, eval_dataset)

    # Testing
    if args.do_test and args.local_rank in [-1, 0]:
        test_dataset = RetrievalDataset(args, 'test', tokenizer, is_train=False)
        evaluate(args, model, test_dataset)


if __name__ == "__main__":
    main()
