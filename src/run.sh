#!/bin/bash

# Define the arguments for the Python script
DATA="data/openi_RN50_train.pkl"
OUT_DIR="./outputs"
PREFIX="openi_prefix"
EPOCHS=10
SAVE_EVERY=1
PREFIX_LENGTH=10
PREFIX_LENGTH_CLIP=10
BS=40
ONLY_PREFIX=false
MAPPING_TYPE="mlp"
NUM_LAYERS=8
IS_RN=false
NORMALIZE_PREFIX=false

# Run the Python script with the defined arguments
python3 train.py \
    --data "$DATA" \
    --out_dir "$OUT_DIR" \
    --prefix "$PREFIX" \
    --epochs "$EPOCHS" \
    --save_every "$SAVE_EVERY" \
    --prefix_length "$PREFIX_LENGTH" \
    --prefix_length_clip "$PREFIX_LENGTH_CLIP" \
    --bs "$BS" \
    --only_prefix "$ONLY_PREFIX" \
    --mapping_type "$MAPPING_TYPE" \
    --num_layers "$NUM_LAYERS" \
    --is_rn "$IS_RN" \
    --normalize_prefix "$NORMALIZE_PREFIX"