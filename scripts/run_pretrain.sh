#!/bin/bash
export STORE_DIR=XXX
# trainer of transformers will automatically call all 
# available GPUs for data paralle training
export CUDA_VISIBLE_DEVICES=0

python pretrain.py \
    --model_name_or_path $STORE_DIR/model/bert-base-chinese \
    --data_path $STORE_DIR/data/XXX \
    --graph_vocab_path $STORE_DIR/node_vocab.txt \
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
