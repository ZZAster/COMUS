from datasets import load_from_disk

import logging
import sys
import os
import json

import transformers

from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AdamW,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from model.data_collator import DataCollatorForGraphPreTrain
from model.config import GATBertConfig
from model.comus import BertForPreTraining


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    g_num_layers: int = field(
        default=6,
        metadata={
            "help": "Num of Gat Layer."
        },
    )
    tokenizer_path: str = field(
        default=None,
        metadata={
            "help": "The tokenizer checkpoint."
            "Default to model_name_or_path if not set."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(
        default=None,
        metadata={"help": "The input training data."}
    )
    add_token_path: str = field(
        default=None,
        metadata={"help": "The additional tokens of your corpus. (a json file that stores a word list)"}
    )
    graph_vocab_path: str = field(
        default=None,
        metadata={"help": "The graph vocab data file (a json file that stores a node list)."}
    )
    max_input_length: int = field(
        default=256,
        metadata={"help": "The max length for text input ids."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for maksed language modeling loss"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    with open(data_args.graph_vocab_path) as f:
        graph_vocab = json.load(f)

    model_path = model_args.model_name_or_path
    tokenizer_path = model_args.tokenizer_path if model_args.tokenizer_path else model_args.model_name_or_path
    config = GATBertConfig.from_pretrained(model_path, g_num_layers=model_args.g_num_layers, g_in_dim=len(graph_vocab))
    
    model = BertForPreTraining.from_pretrained(model_path, config=config)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if data_args.add_token_path:
        with open(data_args.add_token_path) as f:
            tokenizer.add_tokens(json.load(f))
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_from_disk(data_args.data_path)
    
    max_length = min(512, data_args.max_input_length)
    
    def process(examples):
        results = tokenizer(examples['content'], max_length=max_length, padding=True, truncation=True)
        for k in ['node_ids', 'rel_ids', 'src', 'dst']:
            results[k] = examples[k]
        return results
    
    train_dataset = train_dataset.map(process, batched=True)

    data_collator = DataCollatorForGraphPreTrain(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    new_module = ['gat', 'rel']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_module)],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_module)],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    logger.info("*** Train ***")

    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    train_result = trainer.train()
    trainer.save_model()
    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == '__main__':
    main()