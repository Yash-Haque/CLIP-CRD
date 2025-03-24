#!/bin/bash

# Move to the project root directory
cd "$(dirname "$0")/.." || exit 1

# Define paths
SRC_DIR="./src"
DATA_DIR="./data"
OUTPUTS_DIR="./outputs"

# Run the script with arguments
python "$SRC_DIR/parse_openi.py" --clip_model_type "RN50"