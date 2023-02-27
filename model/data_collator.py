# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

import torch

import dgl
from dgl.dataloading import batch as batch_graph
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


InputDataClass = NewType("InputDataClass", Any)


@dataclass
class DataCollatorForGraph:

    label_name: str
    g_in_dim: int
    tokenizer: PreTrainedTokenizerBase
    need_graph: bool

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch['input_ids'] = torch.tensor([e['text_id'] for e in examples])
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch["attention_mask"][batch["input_ids"] == self.tokenizer.pad_token_id] = 0

        if self.need_graph:
            unbatched_g = [dgl.graph((e["src"], e["dst"]), num_nodes=len(e["node_id"])) for e in examples]
            # unbatched_g = [dgl.add_self_loop(g) for g in unbatched_g]
            g = batch_graph(unbatched_g)
            g.ndata["node_ids"] = torch.tensor(sum([e["node_id"] for e in examples], []))
            if 'rel_id' in examples[0]:
                g.edata['rel_ids'] = torch.tensor(sum([e["rel_id"] for e in examples], []))
            batch["graph"] = g

        batch["labels"] = torch.tensor([e[self.label_name] for e in examples])

        return batch


@dataclass
class DataCollatorForGraphPreTrain:

    tokenizer: PreTrainedTokenizerBase
    graph_vocab: dict
    mlm_probability: float = 0.15

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        # input_keys = ["input_ids", "attention_mask", "graph", "ccp_label"]
        batch = {}
        batch['input_ids'] = torch.tensor([e['input_ids'] for e in examples])
        batch['token_type_ids'] = torch.tensor([e['token_type_ids'] for e in examples])
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch["attention_mask"][batch["input_ids"] == self.tokenizer.pad_token_id] = 0

        # prepare graph
        unbatched_g = [dgl.graph((e["src"], e["dst"]), num_nodes=len(e["nodes_id"])) for e in examples]
        g = batch_graph(unbatched_g)
        g.edata['rel_ids'] = torch.tensor(sum([e["rel_id"] for e in examples], []))
        g.ndata['node_ids'] = torch.tensor(sum([e["nodes_id"] for e in examples], []))
        batch["graph"] = g

        # prepare mask tokens and labels
        special_tokens_mask = (batch['input_ids'] == self.tokenizer.sep_token_id) | (batch['input_ids'] == self.tokenizer.cls_token_id)
        batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"], special_tokens_mask)

        return batch


    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special token mask
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
