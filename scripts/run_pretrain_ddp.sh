#!/bin/bash
export STORE_DIR=XXX

python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
    --model_name_or_path $STORE_DIR/ckpt/bert-base-chinese \
    --data_path $STORE_DIR/data/graph_dataset \
    --graph_vocab_path $STORE_DIR/data/graph_vocab.json \
    --seed 2021 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --save_steps 20000 \
    --logging_steps 20 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_steps 100000 \
    --output_dir $STORE_DIR/results/comus \
