#!/bin/bash
# Inference Script
# Author: Philip Chikontwe

echo "Model Inference ..."
# path/to/
# Root directory containing WSI feature files (.pt/.pth)
WSI_ROOT="" #"./examples/data/wsi_features"
FEATURE="uni"
WSI_ROOT="..path/to/.../dfcibrain_sparse/${FEATURE}/features"

PATIENT_CSV="..path/to/.../csv/dfcibrain_coarse_clean.csv"
# Split file CSV ROOT DIR (with train, val, test columns containing case_ids)
# Contains split_0.csv, split_1.csv, etc.
SPLIT_CSV="..path/to/.../datasets/5foldcv_/dfcibrain_coarse"

# Pretrained model checkpoint (directory or file)
CHECKPOINT="./pretrained/weights/brainNIH/${FEATURE}"

# Concept embeddings root directory
CONCEPT_ROOT="./pretrained/concepts"

# Output directory for results
OUTPUT_DIR="./results"

# Model parameters
PREV_TYPE="brainNIH"
TEXT_MODEL="cosmo"
NUM_WORKERS=4

# Inference parameters
RUNS=1 # default 5
EVAL_ALL="--eval-all"

echo ""
echo "Inference with pretrained model"
echo "---------------------------------------------"

python test.py \
    --wsi-root $WSI_ROOT \
    --patient-csv $PATIENT_CSV \
    --split-csv $SPLIT_CSV \
    --checkpoint $CHECKPOINT \
    --prev-type $PREV_TYPE \
    --text-model $TEXT_MODEL \
    --concept-root $CONCEPT_ROOT \
    --output-dir $OUTPUT_DIR \
    --num-workers $NUM_WORKERS \
    --runs $RUNS \
    $EVAL_ALL

