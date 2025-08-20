#!/bin/bash
# Test WSI Bag Dataset functionality
# Author: Philip Chikontwe

echo "Testing COSMO WSI Bag Dataset..."

# =============================================================================
# CONFIGURATION - Fill in your actual paths
# =============================================================================

# Root directory containing WSI feature files (.pt/.pth)
WSI_ROOT="..path/to/.../data/dfcibrain_sparse/uni/features"

# Patient info CSV file (with case_id, slide_id, class_name columns)
PATIENT_CSV="..path/to/.../datasets/csv/dfcibrain_coarse_clean.csv"

# Split file CSV (with train, val, test columns containing case_ids)
SPLIT_CSV="..path/to/.../datasets/5foldcv_/dfcibrain_coarse/splits_0.csv"

# Template paths for knowledge (optional - can be empty dict)
TEMPLATES_DIR="./src/cosmo/data/knowledge/templates"

# =============================================================================
# TEST PARAMETERS
# =============================================================================

# Dataset parameters
PREV_TYPE="brainNIH"
BATCH_SIZE=1
NUM_WORKERS=4
WSI_TOKENS=256

# Test modes
TEST_MODES="train val test"
ZS_STATES="seen unseen"