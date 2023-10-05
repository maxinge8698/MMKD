# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from .modeling_bert import (BertEmbeddings,
                            BertSelfAttention, BertSelfOutput, BertAttention,
                            BertIntermediate, BertOutput,
                            BertLayer,
                            BertEncoder,
                            BertModel,
                            BertPooler,
                            BertLayerNorm,
                            BertPreTrainedModel,)

logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for history_state.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores: (n×d/h)(d/h×n)=(n×n)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # (n×n) / sqrt(h)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Softmax(QK^T/sqrt(d_k))V: (n×d/h)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for history_state.
    """

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)

        self.self = CaptionBertSelfAttention(config)  # 修改BertSelfAttention(nn.Module)为CaptionBertAttention(BertSelfAttention)
        self.output = BertSelfOutput(config)

    def forward(self,
                input_tensor,
                attention_mask,
                head_mask=None,
                history_state=None):
        self_outputs = self.self(input_tensor,
                                 attention_mask,
                                 head_mask,
                                 history_state)
        attention_output = self.output(self_outputs[0],
                                       input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for history_state.
    """

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)

        self.attention = CaptionBertAttention(config)  # 修改BertAttention(nn.Module)为CaptionBertAttention(BertAttention)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states,
                                           attention_mask,
                                           head_mask,
                                           history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for encoder_history_states.
    """

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])  # 修改BertLayer(nn.Module)为CaptionBertLayer(BertLayer)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]

            layer_outputs = layer_module(hidden_states,
                                         attention_mask,
                                         head_mask[i],
                                         history_state)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertImgModel(BertPreTrainedModel):
    """
    Expand from BertModel to handle image region features as input
    """

    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)  # 修改BertEncoder(nn.Module)为CaptionBertEncoder(BertEncoder)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type

        # if hasattr(config, 'use_img_layernorm'):
        #     self.use_img_layernorm = config.use_img_layernorm
        # else:
        #     self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size)  # 2054维转768维
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 为img_embedding_output新增dropout
        # if self.use_img_layernorm:
        #     self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,  # (32,128)
                token_type_ids=None,  # (32,128)
                attention_mask=None,  # (32,178)
                img_feats=None,  # (32,50,2054)
                position_ids=None,
                head_mask=None,
                encoder_history_states=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (32,1,1,178)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers  # [None, None, None, None, None, None, None, None, None, None, None, None]

        embedding_output = self.embeddings(input_ids,
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids)  # (32,128,768)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)  # (32,50,2024) -> (32,50,768)
            # if self.use_img_layernorm:
            #     img_embedding_output = self.LayerNorm(img_embedding_output)
            # add dropout on image embedding
            img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)  # ((32,128,768), (32,50,768), dim=1) -> (32,178,768)

        encoder_outputs = self.encoder(embedding_output,  # (32,178,768)
                                       extended_attention_mask,  # (32,1,1,178)
                                       head_mask=head_mask,  # [None, None, None, None, None, None, None, None, None, None, None, None]
                                       encoder_history_states=encoder_history_states)

        sequence_output = encoder_outputs[0]  # (32,178,768)
        pooled_output = self.pooler(sequence_output)  # (32,768)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


class OscarForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config, fit_size=768):
        super(OscarForSequenceClassification, self).__init__(config)

        self.config = config

        self.num_labels = config.num_labels
        self.loss_type = config.loss_type

        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2
            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        #
        if config.output_hidden_states and (config.hidden_size < fit_size):  # 312 < 768
            self.need_transform = True
            self.fit_dense = nn.Linear(config.hidden_size, fit_size)  # 312维转768维
        else:
            self.need_transform = False
        #

        self.apply(self.init_weights)

    def forward(self,
                input_ids,  # (32,128)
                token_type_ids=None,  # (32,128)
                attention_mask=None,  # (32,178)
                labels=None,  # (16,3129)或(16,2)
                img_feats=None,  # (16,50,2054)
                position_ids=None,
                head_mask=None):
        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(input_ids,  # (32,128)
                                token_type_ids=token_type_ids,  # (32,128)
                                attention_mask=attention_mask,  # (32,178)
                                img_feats=img_feats,  # (32,50,2054)
                                position_ids=position_ids,  # None
                                head_mask=head_mask)  # None
        else:
            outputs = self.bert(input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                head_mask=head_mask)

        pooled_output = outputs[1]  # (32,768)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # (32,3129)或(32,2)

        # add hidden states and attention if they are here
        if self.need_transform:
            all_hidden_states = outputs[2]  # tuple(13个(16, 178, 312))
            all_attentions = outputs[3]  # tuple(12个(16, 12, 178, 178))
            tmp = []
            for i, hidden_states in enumerate(all_hidden_states):
                tmp.append(self.fit_dense(hidden_states))  # (16, 178, 312)转(16,178,768)
            outputs = (logits,) + (tmp, all_attentions)
        else:
            outputs = (logits,) + outputs[2:]
        #

        if labels is not None:
            if self.num_labels == 1:  # doing regression
                loss_fct = nn.MSELoss(reduction='mean')
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = nn.KLDivLoss(reduction="batchmean")
                    log_softmax = nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce':  # [VQA]
                    # loss = instance_bce_with_logits(logits, labels)
                    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
                    loss = loss_fct(logits, labels)  # BCE((32,3129), (32,3129)) -> torch(数)
                    loss *= labels.size(1)  # torch(数) * 3129
                elif self.loss_type == 'ce':  # [Retrieval]
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # CE((32,2), (32,)) -> torch(数)
                else:
                    raise NotImplementedError()
            outputs = (loss,) + outputs  # loss, hidden_states, attentions

        return outputs


class OscarForMultipleChoice(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """

    def __init__(self, config, fit_size=768):
        super(OscarForMultipleChoice, self).__init__(config)

        self.config = config

        self.num_labels = config.num_labels
        self.loss_type = config.loss_type

        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2
            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.num_choice * config.hidden_size, config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.num_choice * config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.num_choice * config.hidden_size, config.num_labels)

        #
        if config.output_hidden_states and (config.hidden_size < fit_size):  # 312 < 768
            self.need_transform = True
            self.fit_dense = nn.Linear(config.hidden_size, fit_size)  # 312维转768维
        else:
            self.need_transform = False
        #

        self.apply(self.init_weights)

    def forward(self,
                input_ids,  # (32,2,55)
                token_type_ids=None,  # (32,2,55)
                attention_mask=None,  # (32,2,95)
                labels=None,  # (32,1)
                img_feats=None,  # (32,2,50,2054)
                position_ids=None,
                head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # (32*2,55)
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None  # (32*2,55)
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None  # (32*2,95)
        flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None  # (32*2,50,2054)
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None  # None

        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(flat_input_ids,  # (32*2,55)
                                token_type_ids=flat_token_type_ids,  # (32*2,55)
                                attention_mask=flat_attention_mask,  # (32*2,95)
                                img_feats=flat_img_feats,  # (32*2,50,2054)
                                position_ids=flat_position_ids,  # None
                                head_mask=head_mask)  # None
        else:
            outputs = self.bert(flat_input_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask,
                                position_ids=flat_position_ids,
                                head_mask=head_mask)

        pooled_output = outputs[1]  # (32*2,768)

        pooled_output = self.dropout(pooled_output)  # (32*2,768)

        reshaped_pool_output = pooled_output.view(-1, self.config.num_choice * (pooled_output.shape[1]))  # (32,2*768)

        logits = self.classifier(reshaped_pool_output)  # (32,2)

        # add hidden states and attention if they are here
        if self.need_transform:
            all_hidden_states = outputs[2]  # tuple(13个(16, 178, 312))
            all_attentions = outputs[3]  # tuple(12个(16, 12, 178, 178))
            tmp = []
            for i, hidden_states in enumerate(all_hidden_states):
                tmp.append(self.fit_dense(hidden_states))  # (16, 178, 312)转(16,178,768)
            outputs = (logits,) + (tmp, all_attentions)
        else:
            outputs = (logits,) + outputs[2:]
        #

        if labels is not None:
            if self.loss_type == 'ce':  # [NLVR]
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(logits, labels.view(-1))  # CE((32,2), (32,)) -> torch(数)
            else:
                raise NotImplementedError()
            outputs = (loss,) + outputs  # loss, hidden_states, attentions

        return outputs
