#!/bin/bash

INPUT_SUBDIR="experiments"
EMBEDDING_DIM=300
VOCAB_SIZE=10000
WINDOW_SIZE=3
BATCH_SIZE=128
EPOCHS=10
NUM_THREADS=16
NOTE="_"

python3 main.py --input_subdir $INPUT_SUBDIR \
                 --embedding_dim $EMBEDDING_DIM \
                 --vocab_size $VOCAB_SIZE \
                 --window_size $WINDOW_SIZE \
                 --batch_size $BATCH_SIZE \
                 --epochs $EPOCHS \
                 --num_threads $NUM_THREADS \
                 --note $NOTE