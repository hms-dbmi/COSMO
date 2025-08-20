#!/bin/bash

DATASET="brain" # lung, kidney
TXT_MODEL="cosmo"
CONCEPTS_JSON="./pretrained/concepts"
TEMPLATES="./src/cosmo/data/knowledge/templates"
OUTPUT_DIR="./examples/concepts"
COSMOLM_CHECKPOINT="./pretrained/weights/cosmollm"


CUDA_VISIBLE_DEVICES="0"  python scripts/extract_embeddings.py \
    --dataset $DATASET \
    --text-model $TXT_MODEL \
    --prev-type "${DATASET}NIH" \
    --output-dir $OUTPUT_DIR \
    --concepts-dir $CONCEPTS_JSON \
    --templates-dir $TEMPLATES \
    --checkpoint $COSMOLM_CHECKPOINT