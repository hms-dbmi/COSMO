# COSMO: slide-level predictions, patient-level splits and demo data

Data record accompanying "Biomedical Knowledge-Enhanced Multimodal Learning for Common and Rare Cancer
Pathology Diagnosis - A Multi-Center Validation Study" (Chikontwe et al.). Code:
https://github.com/hms-dbmi/COSMO (archived at 10.5281/zenodo.CODE_ID).

## Contents

| Folder | What it holds |
|---|---|
| `predictions/` | One CSV per method, foundation model, training setting (full-shot, few-shot), cohort and split, as written by `test.py`: `slide, label, <one softmax column per class>`. Covers COSMO and all baselines on the DFCI test sets and the external cohorts (MUV, TVGH, HKU, TCGA), plus the brain mimic and benign cohorts used for the uncertainty analyses. |
| `splits/` | The five patient-level DFCI splits (`splits_{0-4}.csv`; columns train, val, test). |
| `slide_to_patient/` | Slide-to-patient map for each cohort, used for patient-level bootstrap CIs. |
| `demo/` | Tile features (UNI, GigaPath, CHIEF; CLAM .h5 with `features` and `coords`) and expected outputs for the public TCGA demo slides, including one slide outside the training scope. |
| `Data_S1_Source_Data.xlsx` | Values plotted in every graph of the paper, one tab per figure. |
| `MANIFEST.json` | File list with slide counts and class names. |

## De-identification

Slide and patient identifiers from DFCI, MUV, TVGH, HKU and HistAI were replaced by salted SHA-256
hashes (`S` + 16 hex characters). The same slide has the same hash in every file, so predictions can be
joined across methods, foundation models and splits. TCGA identifiers are public and kept as is. The
record contains no images (other than tile features of public TCGA slides), dates, ages, sex or
free-text diagnoses.

## Reproducing the reported metrics

```bash
git clone https://github.com/hms-dbmi/COSMO && cd COSMO
python release/compute_metrics.py --files "<record>/predictions/**/*.csv" \
    --slide-to-patient "<record>/slide_to_patient/*.csv" --out metrics.csv
# mean-softmax ensemble over foundation models for one cohort and split, as in the main results
# (adjust the glob to the folder names in predictions/, see MANIFEST.json)
python release/compute_metrics.py --ensemble --files "<record>/predictions/cosmo/*/full/dfcibrain/predictions_*_0_all.csv"
```

## Access to images

TCGA slides: GDC portal (https://portal.gdc.cancer.gov). MUV brain cohort: EBRAINS
(https://doi.org/10.25493/WQ48-ZGX). Whole-slide images from DFCI, MUV lung, TVGH and HKU cannot be
shared publicly under the approved protocols and are available from the lead contact (Kun-Hsing Yu,
Kun-Hsing_Yu@hms.harvard.edu) under institutional data use agreements.

## License

CC BY-NC-ND 4.0, non-commercial academic research use with attribution.
