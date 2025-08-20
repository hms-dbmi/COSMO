
## 1. TEST ONCOTREE Extractor

# ##BRAIN
python ./src/cosmo/data/knowledge/oncotree_extractor.py --tissue BRAIN --output ./examples/data/brain_KT_onco.json
# ##LUNG
python ./src/cosmo/data/knowledge/oncotree_extractor.py --tissue LUNG --output ./examples/data/lung_KT_onco.json
# ##KIDNEY
python ./src/cosmo/data/knowledge/oncotree_extractor.py --tissue KIDNEY --output ./examples/data/kidney_KT_onco.json

## 2. TEST NCI Enrichment

# ## Enrich brain knowledge
python ./src/cosmo/data/knowledge/nci_extractor.py --input ./examples/data/brain_KT_onco.json --output ./examples/data/brain_KT_nci.json

## Enrich lung knowledge
python ./src/cosmo/data/knowledge/nci_extractor.py --input ./examples/data/lung_KT_onco.json --output ./examples/data/lung_KT_nci.json

## Enrich kidney knowledge
python ./src/cosmo/data/knowledge/nci_extractor.py --input ./examples/data/kidney_KT_onco.json --output ./examples/data/kidney_KT_nci.json


# 3. TEST UMLS Enrichment (Requires API Key)

# NOTE: Replace YOUR_UMLS_API_KEY with your actual UMLS API key from https://uts.nlm.nih.gov/uts/
UMLS_API_KEY="YOUR API KEY"

# Enrich brain knowledge with UMLS
python ./src/cosmo/data/knowledge/umls_extractor.py --input ./examples/data/brain_KT_nci.json --output ./examples/data/brain_KT_umls.json --api-key $UMLS_API_KEY

# Enrich lung knowledge with UMLS
python ./src/cosmo/data/knowledge/umls_extractor.py --input ./examples/data/lung_KT_nci.json --output ./examples/data/lung_KT_umls.json --api-key $UMLS_API_KEY

# Enrich kidney knowledge with UMLS
python ./src/cosmo/data/knowledge/umls_extractor.py --input ./examples/data/kidney_KT_nci.json --output ./examples/data/kidney_KT_umls.json --api-key $UMLS_API_KEY


# 4. TEST Histocytological Features (PathologyOutlines.com scraping)
# Needs lot of manual work.
# IMPORTANT: use appropriate delays

# Enrich brain knowledge with pathology features 
#python ./src/cosmo/data/knowledge/histcyto_extractor.py --input ./examples/data/brain_KT_umls.json --output ./examples/data/brain_KT_path.json #--delay 2.0

# Enrich lung knowledge with pathology features
#python ./src/cosmo/data/knowledge/histcyto_extractor.py --input ./examples/data/lung_KT_umls.json --output ./examples/data/lung_KT_path.json #--delay 2.0

# Enrich kidney knowledge with pathology features 
#python ./src/cosmo/data/knowledge/histcyto_extractor.py --input ./examples/data/kidney_KT_umls.json --output ./examples/data/kidney_KT_path.json #--delay 2.0

# Debug mode for testing specific tumors only 
#python ./src/cosmo/data/knowledge/histcyto_extractor.py --input ./examples/data/brain_KT_umls.json --output ./examples/data/brain_KT_path_debug.json --debug --delay 1.0



# 5. GENERATE Training CSV Files

# Generate all CSV types for brain (using UMLS-enriched data as fallback)
python ./src/cosmo/data/knowledge/csv_generator.py --input ./examples/data/brain_KT_umls.json --tissue brain --output-dir ./examples/data/ --csv-types train

# # Generate all CSV types for lung
python ./src/cosmo/data/knowledge/csv_generator.py --input ./examples/data/lung_KT_umls.json --tissue lung --output-dir ./examples/data/ --csv-types train

# # Generate all CSV types for kidney
python ./src/cosmo/data/knowledge/csv_generator.py --input ./examples/data/kidney_KT_umls.json --tissue kidney --output-dir ./examples/data/ --csv-types train

# Generate specific CSV types only (example)
#python ./src/cosmo/data/knowledge/csv_generator.py --input ./examples/data/brain_KT_path.json --tissue brain --output-dir ./examples/data/ --csv-types train 