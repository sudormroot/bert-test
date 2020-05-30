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
export GLUE_DIR="$script_path/glue_data"

rm -rf mrpc_output 2>/dev/zero

stdbuf -oL python $script_path/run_classifier.py  \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=mrpc_output/ | tee $script_path/output_batchsize-4_umem-$UNIFIED_MEMORY_SET.txt

rm -rf mrpc_output 2>/dev/zero

stdbuf -oL python $script_path/run_classifier.py  \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=mrpc_output/ | tee $script_path/output_batchsize-8_umem-$UNIFIED_MEMORY_SET.txt

rm -rf mrpc_output 2>/dev/zero

stdbuf -oL python $script_path/run_classifier.py  \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=mrpc_output/ | tee $script_path/output_batchsize-32_umem-$UNIFIED_MEMORY_SET.txt

