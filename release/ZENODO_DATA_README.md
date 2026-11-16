# COSMO: slide-level predictions, patient-level splits, tile features and demo data

Data record accompanying "Biomedical Knowledge-Enhanced Multimodal Learning for Common and Rare Cancer
Pathology Diagnosis - A Multi-Center Validation Study" (Chikontwe et al.). Code:
https://github.com/hms-dbmi/COSMO. Record: [10.5281/zenodo.24291267](https://doi.org/10.5281/zenodo.24291267).

## Contents

| Folder | What it holds |
|---|---|
| `predictions/` | One CSV per method, foundation model, training setting (full-shot, few-shot), cohort and split, as written by `test.py`: `slide, label, <one softmax column per class>`. Covers COSMO and all baselines on the DFCI test sets and the external cohorts (MUV, TVGH, HKU, TCGA), plus the brain mimic cohorts used for the uncertainty analyses. Paths follow the code: `<cohort>_<ratio>_<label set>/<method>[_none]_<model>_<ratio>/[<training cohort>/]predictions_<method>_test_<run>_<split>.csv`. See *Naming* below. |
| `splits/` | The five patient-level DFCI splits for brain, lung and kidney (`splits_{0-4}.csv`; columns train, val, test). |
| `features_<model>_<cohort>.zip` | CHIEF and UNI tile features, labels and splits for training and testing COSMO (see *Training features*). Checksums in `MD5SUMS.txt`. |
| `slide_to_patient/` | Slide-to-patient map for each cohort, used for patient-level bootstrap CIs. |
| `demo/` | For five public TCGA slides (two lung, two kidney, one breast outside the training scope): tile features (UNI, GigaPath, CHIEF; .h5 with `features` and `coords`, one shared tile grid per slide), slide thumbnails, expected outputs, and `tcga_demo_manifest.tsv` (GDC file IDs and md5) for downloading the .svs files if you want to re-extract features. |
| `Data_S1_Source_Data.xlsx` | Values plotted in every graph of the paper, one tab per figure. |
| `MANIFEST.json` | File list with slide counts and class names; `features` lists each zip with slide, patient and class counts, size and md5. |

## Naming

- **Training setting (`<ratio>`):** `0.16` = full-shot (all training slides), `0.4` = few-shot (4 slides per
  subtype), `1.0` = zero-shot (no training on our cohorts). The ratio is the value of the code's `--ratio`
  argument, kept so that files match the code.
- **Method (`<method>`):** `oursmm` = COSMO; `abmil`, `clam`, `meanmil` = MIL baselines; `coop`, `top`,
  `cate`, `mscpt` = vision-language prompt/adapter baselines; `zsvlm` = zero-shot KEEP;
  `prism2` = zero-shot Prism2 (Virchow2 tile features). COSMO is also provided on Virchow2 (`oursmm_none_virchow2_*`)
  for the comparison with Prism2 on the same tile features. Only the evaluations reported in the paper are
  included (no ablations).
- **Run:** `test_<k>` is DFCI split *k* (`splits/splits_<k>.csv`). External cohorts are scored by each run's model.
- **Split:** `seen` = common subtypes, `unseen` = rare subtypes, `all` = all subtypes (generalized zero-shot setting).
- **Training cohort:** for external cohorts, the extra folder (e.g. `dfcilung_coarse/`) names the DFCI cohort the model was trained on.

## Training features (CHIEF and UNI)

Tile features (float32, exactly as used in the paper) from two foundation models, CHIEF (768-d) and UNI
(1024-d), for training and testing COSMO. Files are named `features_<model>_<cohort>.zip`:

| Cohort (`<cohort>`) | Role | Slides | Patients | Splits | CHIEF zip | UNI zip |
|---|---|---|---|---|---|---|
| DFCI brain (`dfcibrain`) | development | 1,010 | 978 | 5 | 4.35 GB | 5.79 GB |
| DFCI lung (`dfcilung`) | development | 992 | 912 | 5 | 4.10 GB | 5.46 GB |
| DFCI kidney (`dfcikidney`) | development | 232 | 230 | 5 | 1.41 GB | 1.88 GB |
| MUV brain / EBRAINS (`ebrainstest`) | external test | 1,611 | 1,498 | none (all test) | 1.85 GB | 2.46 GB |
| TCGA kidney (`tcgakidneytest`) | external test | 895 | 895 | none (all test) | 1.31 GB | 1.74 GB |

Each zip contains `<cohort>/<model>/<slide>.pt` (one tensor `[n_tiles, D]` per slide), `<cohort>/labels.csv`
(`slide_id, case_id, class_name`), `<cohort>/splits/splits_{0-4}.csv` (DFCI only) and `<cohort>/README.txt`.
Identifiers are the same hashes as in `predictions/` and `slide_to_patient/`. Features for the other
cohorts and foundation models are available from the lead contact under data use agreements.

Train on DFCI kidney and test on TCGA kidney (full commands in `scripts/release_train_example.sh` of the
code repository):

```bash
unzip features_chief_dfcikidney.zip -d data && unzip features_chief_tcgakidneytest.zip -d data
# test the released kidney model on DFCI kidney split 0
python test.py --wsi-root data/dfcikidney/chief --patient-csv data/dfcikidney/labels.csv \
    --split-csv data/dfcikidney/splits --checkpoint pretrained/weights/kidneyNIH/chief --prev-type kidneyNIH --eval-all
# train on DFCI kidney split 0 (knowledge folder with kidneyNIH_knowledge_train.csv)
python train.py --prev kidneyNIH --dataset dfcikidney_coarse --feat chief --wsi_root data/dfcikidney/chief \
    --patient_csv data/dfcikidney/labels.csv --split_csv data/dfcikidney/splits --data_path <knowledge dir> \
    --json src/cosmo/data/knowledge/templates/kidneyKT_prompt_names.json --model_name checkpoints/cosmollm
# TCGA kidney is test only: scripts/release_train_example.sh builds its test-only split from labels.csv
```

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
