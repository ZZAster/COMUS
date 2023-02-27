# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

import dgl
from dgl.nn.pytorch import GATConv

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.modeling_outputs import SequenceClassifierOutput
from model.config import GATBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "GATBertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
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

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BidiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(2 * config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        # x -> (batch, seq_len, hidden)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # (batch, seq_len, num_heads, head_size)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_size)

    def forward(
        self,
        query,
        key,
        value,
        value_attention_mask=None,
        query_attention_mask=None
    ):
        query_layer = self.transpose_for_scores(self.query(query))
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, num_heads, seq_len_1, head_size) * (batch, num_heads, head_size, seq_len_2)
        # -> (batch, num_heads, seq_len_1, seq_len_2)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        def attention_help(attention, value, attention_mask):
            attention = attention + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer
        
        query_context = attention_help(attention_scores, value_layer, value_attention_mask)
        value_context = attention_help(attention_scores.transpose(-1, -2), query_layer, query_attention_mask)

        return query_context, value_context


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RelAwareCross(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BidiAttention(config)
        self.output = BertSelfOutput(config)
        self.W1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, query, key, value, value_attention_mask, query_attention_mask):
        query_output, value_output = self.attention(query, key, value, value_attention_mask, query_attention_mask)
        query = self.output(query_output, query)
        theta = torch.sigmoid(self.W1(value_output) + self.W2(value))
        value = theta * value_output + (1 - theta) * value
        value = self.LayerNorm(value)
        return query, value


class RelAwareMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_layer = RelAwareCross(config)
    
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def memory_transform(self, rel_embedding_output, graph_output, g):
        rel_len = list(g.batch_num_edges())
        src, dst = g.edges()
        device = graph_output.device

        src = graph_output[src]
        key = torch.cat([src, rel_embedding_output], dim=1)
        dst = graph_output[dst]

        key = torch.split(key, rel_len, dim=0)
        dst = torch.split(dst, rel_len, dim=0)

        max_rel_len = max(rel_len)

        key = torch.stack(
            [torch.cat([h, torch.zeros(max_rel_len - h.size(0), h.size(1), device=device)], dim=0) for h in key]
        )
        dst = torch.stack(
            [torch.cat([h, torch.zeros(max_rel_len - h.size(0), h.size(1), device=device)], dim=0) for h in dst]
        )

        # (batch, seq_len_2) -> (batch, 1, 1, seq_len_2)
        cross_attention_mask = torch.tensor([[1]*l + [0]*(max_rel_len-l) for l in rel_len], device=device)
        cross_attention_mask = self.get_extended_attention_mask(cross_attention_mask)

        return key, dst, cross_attention_mask
    
    def memory_update(self, value, graph_output, g):
        rel_len = list(g.batch_num_edges())
        src, dst = g.edges()
        group_dst = {idx.item(): [] for idx in dst}
        for i, idx in enumerate(dst):
            group_dst[idx.item()].append(i)
        key_dst = sorted(group_dst)
        val_dst = [group_dst[k] for k in key_dst]
        value = torch.cat([h[:l] for h, l in zip(value, rel_len)], dim=0)
        val_dst = torch.stack([value[indices].mean(dim=0) for indices in val_dst])
        mask = torch.zeros(graph_output.shape, device=graph_output.device, dtype=torch.bool)
        mask[key_dst] = True
        graph_output = graph_output.masked_scatter(mask, val_dst)
        return graph_output
    
    def forward(self, extended_attention_mask, query, rel_embedding_output, graph_output, g):
        key, value, cross_attention_mask = self.memory_transform(rel_embedding_output, graph_output, g)
        query, value = self.cross_layer(query, key, value, cross_attention_mask, extended_attention_mask)
        graph_output = self.memory_update(value, graph_output, g)
        return graph_output, query


class GATEmbeddings(nn.Module):
    def __init__(self, config):
        super(GATEmbeddings, self).__init__()
        self.in_dim = config.g_in_dim
        self.num_hidden = config.g_num_hidden
        self.heads = config.g_heads
        self.activation = F.elu
        self.feat_drop = config.g_feat_drop
        self.attn_drop = config.g_attn_drop
        self.negative_slope = config.g_negative_slope

        self.gat_layer = GATConv(
            self.in_dim, self.num_hidden, self.heads, self.feat_drop, self.attn_drop, 
            self.negative_slope, False, self.activation
        )

    def forward(self, g, node_ids):
        inputs = F.one_hot(node_ids, num_classes=self.in_dim).float()
        return self.gat_layer(g, inputs).flatten(1)


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.num_hidden = config.g_num_hidden
        self.heads = config.g_heads
        self.activation = F.elu
        self.feat_drop = config.g_feat_drop
        self.attn_drop = config.g_attn_drop
        self.negative_slope = config.g_negative_slope
        self.residual = config.g_residual

        self.gat_layer = GATConv(
            self.num_hidden * self.heads, self.num_hidden, self.heads, self.feat_drop, self.attn_drop, 
            self.negative_slope, self.residual, self.activation
        )

    def forward(self, g, inputs):
        return self.gat_layer(g, inputs).flatten(1)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


@dataclass
class CrossModelOutputWithPastAndCrossAttentions(ModelOutput):
    
    last_hidden_state: torch.Tensor = None
    node_hidden_state: torch.Tensor = None
    rel_embedding_output: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.embedding = GATEmbeddings(config)
        self.gat_layer = nn.ModuleList([GATLayer(config) for _ in range(config.g_num_layers)]) if config.g_num_layers > 0 else None
        self.rel_embeddings = nn.Embedding(43, config.hidden_size)
        self.rel_cross_layer = nn.ModuleList([RelAwareMemory(config) for _ in range(2)])
        # self.last_layer = BertAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        g=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        rel_embedding_output = self.rel_embeddings(g.edata['rel_ids'])

        self_loop_g = dgl.add_self_loop(g)
        g_hidden_states = self.embedding(self_loop_g, g.ndata['node_ids'])
        for i in range(self.config.g_num_layers - 1):
            g_hidden_states = self.gat_layer[i](self_loop_g, g_hidden_states)

        for i in range(self.config.num_hidden_layers - 1):
            layer_outputs = self.layer[i](hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        
        g_hidden_states, hidden_states = self.rel_cross_layer[0](attention_mask, hidden_states, rel_embedding_output, g_hidden_states, g)

        if self.gat_layer is not None: 
            g_hidden_states = self.gat_layer[-1](self_loop_g, g_hidden_states)
        hidden_states = self.layer[-1](hidden_states, attention_mask)[0]

        g_hidden_states, hidden_states = self.rel_cross_layer[1](attention_mask, hidden_states, rel_embedding_output, g_hidden_states, g)

        # hidden_states = self.last_layer(hidden_states, attention_mask)[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return CrossModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            node_hidden_state=g_hidden_states,
            rel_embedding_output=rel_embedding_output,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class GraphPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = 2
        self.margin = 1.0
        self.distfn = nn.PairwiseDistance(self.L)
    
    def memory_transform(self, g, graph_output, rel_embedding_output):
        pos_src, pos_dst = g.edges()
        mask = g.edata['rel_ids'] == 40
        pos_src, pos_dst = pos_src[~mask], pos_dst[~mask]
        neg_src, neg_dst = self.neg_sample(pos_src, pos_dst)

        pos_src, pos_dst = graph_output[pos_src], graph_output[pos_dst]
        neg_src, neg_dst = graph_output[neg_src], graph_output[neg_dst]
        rel = rel_embedding_output[~mask]

        return (pos_src, rel, pos_dst), (neg_src, rel, neg_dst)

    def neg_sample(self, src, dst):
        neg_src = src.clone()
        neg_dst = dst.clone()
        neg_dst = neg_dst[torch.randperm(neg_dst.size(0))]
        return neg_src, neg_dst

    def scoreOp(self, inputTriple):
        head, relation, tail = inputTriple
        output = self.distfn(head+relation, tail)
        return output

    def forward(self, g, graph_output, rel_embedding_output):
        pos_input, neg_input = self.memory_transform(g, graph_output, rel_embedding_output)
        size = pos_input[0].size(0)
        posScore = self.scoreOp(pos_input)
        negScore = self.scoreOp(neg_input)
        return torch.sum(F.relu(input=posScore-negScore+self.margin)) / size


class GraphContrastiveLearningHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config.temperature
    
    def forward(self, graph_pooled, bert_pooled):
        
        if dist.is_initialized() and self.training:

            graph_pooled_list = [torch.zeros_like(graph_pooled) for _ in range(dist.get_world_size())]
            bert_pooled_list = [torch.zeros_like(bert_pooled) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=graph_pooled_list, tensor=graph_pooled.contiguous())
            dist.all_gather(tensor_list=bert_pooled_list, tensor=bert_pooled.contiguous())


            graph_pooled_list[dist.get_rank()] = graph_pooled
            bert_pooled_list[dist.get_rank()] = bert_pooled
            graph_pooled = torch.cat(graph_pooled_list, 0)
            bert_pooled = torch.cat(bert_pooled_list, 0)

        batch_size = graph_pooled.size(0)
        device = graph_pooled.device
        loss_fct = CrossEntropyLoss()
        features = torch.cat([bert_pooled, graph_pooled], dim=0)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        return loss_fct(logits, labels)


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GATBertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class GATBertOutputWithPoolingAndCrossAttentions(ModelOutput):
    
    node_hidden_state: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    rel_embedding_output: torch.Tensor = None
    graph_pooled: torch.Tensor = None
    bert_pooled: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GATBertForPreTrainingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    node_prediction_logits: torch.FloatTensor = None
    edge_prediction_logits: torch.Tensor = None
    prediction_logits: torch.FloatTensor = None
    ccp_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=GATBertOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            g=graph,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, graph_output, rel_embedding_output = encoder_outputs[:3]  # (batch_size, seq_len, hidden_size)

        bert_pooled = self.pooler(sequence_output) if self.pooler is not None else None
        graph.ndata["h"] = graph_output
        graph_pooled = dgl.mean_nodes(graph, "h")

        if not return_dict:
            return (graph_output, sequence_output, rel_embedding_output, graph_pooled, bert_pooled) + encoder_outputs[1:]

        return GATBertOutputWithPoolingAndCrossAttentions(
            node_hidden_state=graph_output,
            last_hidden_state=sequence_output,
            rel_embedding_output=rel_embedding_output,
            graph_pooled=graph_pooled,
            bert_pooled=bert_pooled,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.graph_cls = GraphPreTrainingHeads(config)
        self.gcl = GraphContrastiveLearningHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device
        graph = graph.to(device)

        outputs = self.bert(
            graph=graph,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        graph_output, sequence_output, rel_embedding_output, graph_pooled, bert_pooled = outputs[:5]
        prediction_scores = self.cls(sequence_output)
        erp_loss = self.graph_cls(graph, graph_output, rel_embedding_output)

        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        gcl_loss = self.gcl(graph_pooled, bert_pooled)

        total_loss = masked_lm_loss + erp_loss + gcl_loss

        if not return_dict:
            output = (prediction_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GATBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.problem_type = config.problem_type

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device
        graph = graph.to(device)

        outputs = self.bert(
            graph,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        graph_pooled, bert_pooled = outputs[3:5]
        pooled_output = torch.cat([bert_pooled, graph_pooled], dim=-1)
        # pooled_output = bert_pooled
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )