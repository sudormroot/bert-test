#!/bin/sh

script_path="`dirname $0`"

. $script_path/env.sh

export CUDA_VISIBLE_DEVICES=1

export UNIFIED_MEMORY_SET=no
#stdbuf -oL python $script_path/train.py --train_batch_size=128 --train_steps=12 2>&1 | tee $script_path/output_batchsize-128_umem-$UNIFIED_MEMORY_SET.txt
#stdbuf -oL python $script_path/train.py --train_batch_size=256 --train_steps=12 2>&1 | tee $script_path/output_batchsize-256_umem-$UNIFIED_MEMORY_SET.txt


export UNIFIED_MEMORY_SET=yes
#BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters

export BERT_BASE_DIR="$script_path/uncased_L-12_H-768_A-12"
export GLUE_DIR="$script_path/glue"

stdbuf -oL python $script_path/run_classifier.py 


#stdbuf -oL python $script_path/train.py --train_batch_size=128 --train_steps=12 2>&1 | tee $script_path/output_batchsize-128_umem-$UNIFIED_MEMORY_SET.txt
#stdbuf -oL python $script_path/train.py --train_batch_size=256 --train_steps=12 2>&1 | tee $script_path/output_batchsize-256_umem-$UNIFIED_MEMORY_SET.txt
