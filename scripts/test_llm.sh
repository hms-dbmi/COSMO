# Test LM training and data loading 

DATA_DIR="./examples/data"
JSON_DIR="./src/cosmo/data/knowledge/templates"
SAVE_PATH="./checkpoints"

# Brain  : [0-6]   7 classes (start at 0)
# Lung   : [7-12]  6 classes (start at index 7)
# Kidney : [13-19] 5 classes (start at index 13)

TISSUES="brain lung kidney"

EPOCHS=1

CUDA_VISIBLE_DEVICES="0" python train_llm.py \
    --data-dir $DATA_DIR \
    --json-dir $JSON_DIR \
    --output-dir $SAVE_PATH \
    --tissues $TISSUES \
    --epochs $EPOCHS