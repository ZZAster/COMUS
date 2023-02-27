from datasets import load_from_disk

import logging
import sys
import os

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
            "Don't set if you want to train a model from scratch."
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
            "Don't set if you want to train a model from scratch."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    graph_vocab_path: str = field(
        default=None,
        metadata={"help": "The graph vocab data file (a text file)."}
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

    graph_vocab = {}
    with open(data_args.graph_vocab_path) as f:
        for word in f:
            graph_vocab[word.strip()] = len(graph_vocab)

    model_path = model_args.model_name_or_path
    config = GATBertConfig.from_pretrained(model_path, g_num_layers=model_args.g_num_layers, g_in_dim=len(graph_vocab))
    
    model = BertForPreTraining.from_pretrained(model_path, config=config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_from_disk(data_args.data_path)

    data_collator = DataCollatorForGraphPreTrain(
        tokenizer=tokenizer, 
        graph_vocab=graph_vocab,
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