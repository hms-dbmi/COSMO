#!/bin/bash

DATASET="dfcibrain"
FEATURE="uni"
# Root directory containing WSI feature files (.pt/.pth)
WSI_ROOT="..path/to/.../data/dfcibrain_sparse/${FEATURE}/features"
# ROOT of {ORGAN}_knowledge_train.csv
KNOWLEDGE="./examples/data"
# Patient info CSV file (with case_id, slide_id, class_name columns)
PATIENT_CSV="..path/to/.../datasets/csv/dfcibrain_coarse_clean.csv"
# Split file CSV ROOT DIR (with train, val, test columns containing case_ids)
# Contains split_0.csv, split_1.csv, etc.
SPLIT_CSV="..path/to/.../datasets/5foldcv_/dfcibrain_coarse"
# Template paths for knowledge (optional - can be empty dict)
# .../{ORGAN}KT_prompt_names.json
JSON_FILE="./src/cosmo/data/knowledge/templates/brainKT_prompt_names.json"
# Prevalence
PREV="brainNIH"

# Number of WSI instances to sample.
TOKENS=1024
TR_RATIOS=(0.16) # (0.2 0.4 0.8 0.16 1.0)
NUM_INSTANCES=4
BATCHSZ=16
DEVICE="cuda"

OUTPUTDIR="./checkpoints/cosmo"
# Folder with PEFT weights for COSMOLM.
PRETRAINED="./checkpoints/cosmollm"
# Folder with class and concept embeddings.
CONCEPTS_DIR="./pretrained/concepts"

echo "Training specialist COSMO ... "

for RUNS in 0; #1 2 3 4;
do

    for TR_RATIO in "${TR_RATIOS[@]}"; #
    do 

        if [ "$TR_RATIO" = 1.0 ]; then
            EPOCHS=80
        else
            EPOCHS=50
        fi
        
        echo "***********************************************************************"
        echo "TR_RATIO : ${TR_RATIO} | EPOCHS: ${EPOCHS} | ${FEATURE} | RUN: ${RUNS}"
        echo "***********************************************************************"

        CUDA_VISIBLE_DEVICES="0" python train.py \
        --model_name $PRETRAINED \
        --concept_root $CONCEPTS_DIR \
        --device $DEVICE --epochs $EPOCHS --output_dir $OUTPUTDIR \
        --wsi_root $WSI_ROOT --json $JSON_FILE \
        --split_csv $SPLIT_CSV --patient_csv $PATIENT_CSV \
        --dataset $DATASET --data_path $KNOWLEDGE --feat $FEATURE \
        --ratio $TR_RATIO --runs $RUNS --tokens $TOKENS --prev $PREV \
        --batch_size $BATCHSZ --num_instances $NUM_INSTANCES

        echo "***********************************************************************"
        echo "RUN: ${RUNS} | COMPLETE!"
        echo "***********************************************************************"

    done 

done 