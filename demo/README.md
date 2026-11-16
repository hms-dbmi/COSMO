# COSMO demo on public TCGA slides

This demo reproduces, for a few public slides, the outputs shown in the paper: the predicted subtype,
attention heatmap, concept activation map, and uncertainty scores, including for a slide from an organ
outside COSMO's training scope.

The pre-extracted tile features and expected outputs for the demo slides are in the Zenodo data record
([10.5281/zenodo.24291267](https://doi.org/10.5281/zenodo.24291267), folder `demo/`). With those files,
step 3 runs on a laptop CPU in under a minute per slide. Steps 1 and 2 are only needed to run COSMO on
your own slides.

| Slide | TCGA project | Role in the demo |
|---|---|---|
| `TCGA-64-1680-01Z-00-DX1` | TCGA-LUAD | lung model, common subtype |
| `TCGA-98-A538-01Z-00-DX1` | TCGA-LUSC | lung model, common subtype |
| `TCGA-B9-4617-01Z-00-DX1` | TCGA-KIRP | kidney model, common subtype |
| `TCGA-KN-8421-01Z-00-DX1` | TCGA-KICH | kidney model, common subtype |
| `TCGA-A1-A0SD-01Z-00-DX1` | TCGA-BRCA (breast) | outside the training scope: scored by all three organ models (`_brain_`, `_lung_`, `_kidney_` outputs) |

(Files are named by the TCGA barcode up to `DX1`; the full GDC file names and UUIDs are in [`tcga_demo_manifest.tsv`](tcga_demo_manifest.tsv).)

## 1. Download the slides (optional)

```bash
# gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
gdc-client download -m demo/tcga_demo_manifest.tsv -d demo/data/slides
```

## 2. Tile the slides and extract features (optional, GPU recommended)

Use [CLAM](https://github.com/mahmoodlab/CLAM) to segment tissue and cut 512 x 512 tiles at 20x
(256 µm), as in the paper, then extract features for each foundation model **from the same tiles**
so that all models share one tile grid:

```bash
python create_patches_fp.py --source demo/data/slides --save_dir demo/data/patches \
    --patch_size 512 --step_size 512 --seg --patch --stitch
# repeat for uni, gigapath and chief (each needs its own weights and licence)
python extract_features_fp.py --data_h5_dir demo/data/patches --data_slide_dir demo/data/slides \
    --csv_path demo/data/patches/process_list_autogen.csv --feat_dir demo/data/features/uni \
    --model_name uni_v1 --batch_size 256 --slide_ext .svs
```

The demo expects `demo/data/features/{uni,gigapath,chief}/{slide_id}.h5`, each with `features` [N, D]
and `coords` [N, 2].

## 3. Run COSMO

```bash
# a lung slide, scored with the lung model
python demo/cosmo_demo.py --slide-id <SLIDE_ID> --features-root demo/data/features \
    --thumbnail demo/data/thumbnails/<SLIDE_ID>.png --organ lung --out demo/outputs   # or --wsi <SLIDE_ID>.svs

# a breast slide: outside the training scope, so score it with all three organ models
python demo/cosmo_demo.py --slide-id <SLIDE_ID> --features-root demo/data/features \
    --wsi demo/data/slides/<SLIDE_ID>.svs --organ auto --out demo/outputs
```

Outputs per slide and organ model (`<slide>_<organ>_*`) and per slide (`<slide>_*`):

- `*_report.json`: subtype probabilities (ensemble and per foundation model), the top concepts with
  their slide-level probability and subtype, and the uncertainty scores (normalized entropy, maximum
  softmax probability, mutual information across foundation models). For `--organ auto`,
  `out_of_scope_scores` averages the three organ models.
- `*_attention.pdf`: attention heatmap (per-tile attention, min-max normalised and averaged over
  foundation models; smoothed jet overlay, background outside the tissue in grey).
- `*_concepts.pdf`: concept activation map; tiles with above-average attention are coloured by their
  top concept, for the 8 most frequent concepts. Legend in `*_concepts_legend.pdf`
  (`(slide-level probability, subtype) concept`); all concepts in `*_concepts.csv`.
- `<slide>_mutual_information.pdf`: one map per slide of the tile-level disagreement between foundation
  models (same styling as the attention map); for `--organ auto` it is averaged over the three organ models.
- `colorbar_attention.pdf`, `colorbar_mutual_information.pdf`: one pair per output folder (maps are
  normalised per slide, so the scale is the same for every slide).

By default each map is normalised per slide; `--mi-vmax 0.7` draws the uncertainty map on a fixed scale
(0–0.7 nats) so that slides can be compared, as in the provided outputs and the README figure
(`demo/make_readme_figure.py`). Maps are 1024 x 1024 images without titles, drawn over the slide (`--wsi`) or the thumbnail
(`--thumbnail`, provided in the Zenodo record).

## How to read the outputs

- **Concept scores** are relative shares that sum to one over the organ's concepts. They show which
  concepts the model associates with a region; they are not the probability that the morphology is
  present, and they should be checked by a pathologist.
- **Uncertainty**. COSMO only knows the subtypes it was trained on. For a slide outside this label
  space, such as a breast carcinoma or benign tissue, it still outputs a subtype, but the entropy is
  higher and the confidence lower than for in-scope slides. In the paper, deferring the most
  uncertain slides improved accuracy on the remaining slides; we recommend reviewing any slide with
  low confidence rather than acting on its predicted subtype.
- The released adapters are the models of split 0 for the vision-only foundation models (UNI,
  GigaPath, CHIEF). The paper averages five splits and six foundation models, so values will differ
  slightly from the figures.

COSMO is a research tool for decision support on slides already diagnosed as malignant brain, lung or
kidney tumors. It is not a medical device.
