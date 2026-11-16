# COSMO: slide-level predictions, patient-level splits and demo data

Data record accompanying "Biomedical Knowledge-Enhanced Multimodal Learning for Common and Rare Cancer
Pathology Diagnosis - A Multi-Center Validation Study" (Chikontwe et al.). Code:
https://github.com/hms-dbmi/COSMO (archived at 10.5281/zenodo.CODE_ID).

## Contents

| Folder | What it holds |
|---|---|
| `predictions/` | One CSV per method, foundation model, training setting (full-shot, few-shot), cohort and split, as written by `test.py`: `slide, label, <one softmax column per class>`. Covers COSMO and all baselines on the DFCI test sets and the external cohorts (MUV, TVGH, HKU, TCGA), plus the brain mimic cohorts used for the uncertainty analyses. Paths follow the code: `<cohort>_<ratio>_<label set>/<method>[_none]_<model>_<ratio>/[<training cohort>/]predictions_<method>_test_<run>_<split>.csv`. See *Naming* below. |
| `splits/` | The five patient-level DFCI splits (`splits_{0-4}.csv`; columns train, val, test). |
| `slide_to_patient/` | Slide-to-patient map for each cohort, used for patient-level bootstrap CIs. |
| `demo/` | For five public TCGA slides (two lung, two kidney, one breast outside the training scope): tile features (UNI, GigaPath, CHIEF; .h5 with `features` and `coords`, one shared tile grid per slide), slide thumbnails, expected outputs, and `tcga_demo_manifest.tsv` (GDC file IDs and md5) for downloading the .svs files if you want to re-extract features. |
| `Data_S1_Source_Data.xlsx` | Values plotted in every graph of the paper, one tab per figure. |
| `MANIFEST.json` | File list with slide counts and class names. |

## Naming

- **Training setting (`<ratio>`):** `0.16` = full-shot (all training slides), `0.4` = few-shot (4 slides per
  subtype), `1.0` = zero-shot (no training on our cohorts). The ratio is the value of the code's `--ratio`
  argument, kept so that files match the code.
- **Method (`<method>`):** `oursmm` = COSMO; `abmil`, `clam`, `meanmil` = MIL baselines; `coop`, `top`,
  `cate`, `mscpt` = vision-language prompt/adapter baselines; `zsvlm` = zero-shot CONCH or KEEP;
  `prism2` = zero-shot Prism2 (Virchow2 tile features). Only the evaluations reported in the paper are
  included (no ablations).
- **Run:** `test_<k>` is DFCI split *k* (`splits/splits_<k>.csv`). External cohorts are scored by each run's model.
- **Split:** `seen` = common subtypes, `unseen` = rare subtypes, `all` = all subtypes (generalized zero-shot setting).
- **Training cohort:** for external cohorts, the extra folder (e.g. `dfcilung_coarse/`) names the DFCI cohort the model was trained on.

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
# mean-softmax ensemble over foundation models for one external cohort, run and split
python release/compute_metrics.py --ensemble \
    --files "<record>/predictions/tcgalungtest_coarse_0.16_lungNIH/oursmm_none_*_0.16/dfcilung_coarse/predictions_oursmm_test_0_all.csv"
```

## Access to images

TCGA slides: GDC portal (https://portal.gdc.cancer.gov). MUV brain cohort: EBRAINS
(https://doi.org/10.25493/WQ48-ZGX). Whole-slide images from DFCI, MUV lung, TVGH and HKU cannot be
shared publicly under the approved protocols and are available from the lead contact (Kun-Hsing Yu,
Kun-Hsing_Yu@hms.harvard.edu) under institutional data use agreements.

## License

CC BY-NC-ND 4.0, non-commercial academic research use with attribution.
