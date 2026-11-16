#!/usr/bin/env python3
"""
COSMO demo: slide prediction, attention heatmap, concept activation map and uncertainty.

Runs the released COSMO adapters on pre-extracted tile features of one slide, for one
organ model (brain, lung or kidney) or, for a slide outside the training scope, for all
three organ models. The maps follow the paper's visualisation code (wsi_vis/vis_utils.py):

  * prediction    : mean-softmax ensemble over foundation models (FMs)
  * attention map : gated-attention score of each tile, min-max normalised per FM and averaged
                    over FMs; Gaussian-smoothed jet overlay (plot_clslevel_map_probs_smooth)
  * concept map   : tiles with above-average attention are kept; for each, the concept
                    probabilities P_j(x) = softmax_j(tau * cos(x, c_j)) are averaged over FMs and
                    the tile is coloured by its top concept; the 8 most frequent concepts are
                    shown, ordered by slide-level probability (plot_conceptlevel_map)
  * uncertainty   : normalised entropy H(p)/log C, maximum softmax probability (MSP) and mutual
                    information (MI) across FMs; for slides outside the training scope the three
                    organ models are averaged (STAR Methods, OOD analysis)
  * MI map        : tile-level MI across FMs, smoothed jet overlay with background greyed out
                    (plot_uncertainty_map styling)

Maps are saved as 1024 x 1024 PDFs without titles; legends and colour bars are separate PDFs.
Concept scores are relative shares that sum to one over the organ's concepts. They are
model-derived associations, not the probability that a feature is present.

Inputs (per FM): an .h5 file with datasets `features` [N, D] and `coords` [N, 2] (level-0 pixel
coordinates; attrs `patch_size` in level-0 pixels and, optionally, `slide_width`/`slide_height`).
Underneath the maps the demo draws the slide (--wsi, .svs) or, without it, a thumbnail (--thumbnail)
scaled with the slide size stored in the .h5 attributes.

Example:
    python demo/cosmo_demo.py --slide-id TCGA-XX-XXXX-01Z-00-DX1 \
        --features-root ./demo/data/features --wsi ./demo/data/slides/TCGA-XX.svs \
        --organ lung --fms uni gigapath chief --out ./demo/outputs

    # slide from an organ outside the training scope (e.g. breast): score with all organs
    python demo/cosmo_demo.py --slide-id TCGA-BRCA-... --organ auto ...
"""
import argparse, csv, json, os, sys
from collections import Counter

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.cosmo.models.llm import MultiModalCLP          # noqa: E402
from src.cosmo.utils.label_mappings import LABEL_MAP    # noqa: E402

ORGANS = {'brain': 'brainNIH', 'lung': 'lungNIH', 'kidney': 'kidneyNIH'}
VISION_FMS = ['uni', 'gigapath', 'chief']   # use the COSMO language-model embeddings
# CONCH adapters were trained with CONCH's own text embeddings; to use them, first run
# scripts/extract_embeddings.py --text-model conch so that {prev}_conch_*.npy exist.

ABBR = {'Adult-type diffuse gliomas': 'ATDG', 'Paediatric-type diffuse low-grade gliomas': 'PDLGG',
        'Circumscribed astrocytic gliomas': 'CAG', 'Ependymal Tumours': 'EPD', 'Meningiomas': 'MNG',
        'Glioneuronal and neuronal tumours': 'GNT', 'Embryonal Tumours': 'EMB',
        'Lung Adenocarcinoma': 'LUAD', 'Lung Squamous Cell Carcinoma': 'LUSC', 'Small Cell Lung Cancer': 'SCLC',
        'Lung Carcinoid': 'LUCAR', 'Lung Neuroendocrine Tumor': 'LUNET', 'Lung Adenosquamous Carcinoma': 'LUAS',
        'Chromophobe Renal Cell Carcinoma': 'CHRCC', 'Papillary Renal Cell Carcinoma': 'PRCC',
        'Renal Clear Cell Carcinoma': 'KIRC', 'Collecting Duct Renal Cell Carcinoma': 'CDRCC', 'Wilms Tumor': 'WT'}

# map styling, as in wsi_vis/vis_utils.py
TOP_K = 8
BRIGHT8 = [(2, 62, 255), (255, 124, 0), (26, 201, 56), (232, 0, 11),        # seaborn "bright", 8 colours
           (139, 43, 226), (159, 72, 0), (241, 76, 193), (163, 163, 163)]
OUT_SIZE = 1024
HEAT_DS, CONCEPT_DS = 128, 32          # render downsample of the smooth heatmaps / concept map
ATT_SIGMA, ATT_ALPHA = 3, 0.45         # plot_clslevel_map_probs_smooth (sigma as in the paper's renders)
UNC_SIGMA, UNC_ALPHA = 3, 0.45         # plot_uncertainty_map
CONCEPT_ALPHA = 0.4                    # plot_conceptlevel_map
GREY_ATT = (220, 220, 220, 255)        # background outside the tissue: attention map (opaque)
GREY_UNC = (220, 220, 220, 255)        # uncertainty map: opaque, like the attention map (vis_utils uses 127, which lets the jet floor tint the background)


# ------------------------------------------------------------------ inputs and model
def load_tiles(path):
    with h5py.File(path, 'r') as f:
        feats = torch.from_numpy(f['features'][:]).float()
        coords = f['coords'][:] if 'coords' in f else None
        attrs = dict(f['coords'].attrs) if coords is not None else {}
    if feats.ndim == 3:
        feats = feats[0]
    return feats, coords, attrs


def load_model(prev, fm, dim, device, concept_root, weights_root, run_id):
    text_model = 'conch' if fm == 'conch' else 'cosmo'
    model = MultiModalCLP(feature_embed_dim=512, visual_dim=dim, device=device, vlm=False,
                          prev_type=prev, text_model=text_model, concept_root=concept_root)
    ckpt = os.path.join(weights_root, prev, fm, f'visual_model_{run_id}.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return model.to(device).eval()


def unique_concepts(model):
    """Concept names can repeat across subtypes; keep the first occurrence (as vis_utils does)."""
    names, keep = [], []
    for i, c in enumerate(model.concept_list):
        if c not in names:
            names.append(c); keep.append(i)
    return names, keep


@torch.no_grad()
def run_fm(model, feats, device, concept_idx):
    x = feats.to(device)
    model.mode = 0
    p_slide = F.softmax(model.inference(x.unsqueeze(0))[0], -1).cpu().numpy()      # [C]
    h = model.visual_projection(x)
    a = model.attention(h)[:, 0]
    a = ((a - a.min()) / (a.max() - a.min() + 1e-12)).cpu().numpy()                   # min-max per slide
    model.mode = 1
    p_tile = F.softmax(model.inference(x), -1).cpu().numpy()                          # [N, C]
    concepts = model.concepts
    model.concepts = concepts[concept_idx]                                            # de-duplicated
    p_concept = F.softmax(model.inference(x, True), -1).cpu().numpy()                # [N, U]
    model.concepts = concepts
    model.mode = 0
    return p_slide, a, p_concept, p_tile


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis)


def uncertainty(p_fms):
    """p_fms: [F, C] slide-level softmax per FM."""
    pbar = p_fms.mean(0)
    C = pbar.shape[-1]
    H = entropy(pbar)
    return {'normalized_entropy': float(H / np.log(C)), 'msp': float(pbar.max()),
            'mutual_information': float(H - entropy(p_fms).mean())}


def concept_owner(model, names, prev):
    """Subtype each concept belongs to (first subtype that lists it)."""
    out = {}
    with open(os.path.join(model.concept_root, f"{prev.replace('NIH', '')}_closed_concepts.json")) as f:
        d = json.load(f)
    for cls, lst in d.items():
        for c in (lst if isinstance(lst, list) else [lst]):
            out.setdefault(c, cls)
    return [ABBR.get(out.get(n, ''), out.get(n, '')) for n in names]


# ------------------------------------------------------------------ rendering (vis_utils recipes)
class Canvas:
    """Background image at a given downsample: the slide (.svs) or a thumbnail scaled to the slide size."""
    def __init__(self, wsi=None, thumbnail=None, dims=None):
        self.wsi, self.thumbnail, self.dims = wsi, thumbnail, dims
        self.slide = None
        if wsi:
            import openslide
            self.slide = openslide.OpenSlide(wsi)
            self.dims = self.slide.dimensions

    def image(self, ds):
        from PIL import Image
        W, H = self.dims
        size = (max(1, int(W / ds)), max(1, int(H / ds)))
        if self.slide is not None:
            lvl = self.slide.get_best_level_for_downsample(ds)
            img = self.slide.read_region((0, 0), lvl, self.slide.level_dimensions[lvl]).convert('RGB')
            arr = np.array(img); arr[arr == 0] = 255
            img = Image.fromarray(arr)
        elif self.thumbnail:
            img = Image.open(self.thumbnail).convert('RGB')
        else:
            img = Image.new('RGB', size, (255, 255, 255))
        return img.resize(size, Image.Resampling.BICUBIC)


def _paint(coords, values, patch_px, ds, shape):
    m = np.zeros(shape, np.float32); cov = np.zeros(shape, bool)
    ps = max(1, int(round(patch_px / ds)))
    for (x, y), v in zip((np.asarray(coords) / ds).astype(int), values):
        m[y:y + ps, x:x + ps] = v; cov[y:y + ps, x:x + ps] = True
    return m, cov


def smooth_map(canvas, coords, values, patch_px, out_pdf, sigma, alpha, grey=None):
    """visualize_categorical_probs_smooth: paint tile values, Gaussian blur, min-max, jet overlay."""
    import matplotlib
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    img = canvas.image(HEAT_DS)
    w, h = img.size
    m, cov = _paint(coords, values, patch_px, HEAT_DS, (h, w))
    if sigma > 0:
        m = gaussian_filter(m, sigma)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heat = matplotlib.colormaps.get_cmap('jet')(m); heat[:, :, 3] = alpha
    out = Image.alpha_composite(img.convert('RGBA'), Image.fromarray((heat * 255).astype(np.uint8)))
    if grey is not None:                                # grey out everything outside the tiled tissue
        g = np.zeros((h, w, 4), np.uint8); g[~cov] = grey
        out = Image.alpha_composite(out, Image.fromarray(g))
    rgb = Image.new('RGB', out.size, (255, 255, 255)); rgb.paste(out.convert('RGB'), (0, 0), out)
    rgb.resize((OUT_SIZE, OUT_SIZE)).save(out_pdf, 'PDF', resolution=600.0)


def concept_map(canvas, coords, labels, patch_px, out_pdf):
    """visualize_categorical: colour each tile by its concept (alpha 0.4, 1-px dark border)."""
    from PIL import Image, ImageOps
    img = np.array(canvas.image(CONCEPT_DS))
    ps = max(1, int(round(patch_px / CONCEPT_DS)))
    for (x, y), k in zip((np.asarray(coords) / CONCEPT_DS).astype(int), labels):
        blk = img[y:y + ps, x:x + ps]
        if blk.size == 0:
            continue
        col = np.ones_like(blk, dtype=np.float32) * np.array(BRIGHT8[k], np.float32)
        blend = (CONCEPT_ALPHA * col + (1 - CONCEPT_ALPHA) * blk).astype(np.uint8)
        blend = np.array(ImageOps.expand(Image.fromarray(blend), border=1, fill=(50, 50, 50))
                         .resize((blk.shape[1], blk.shape[0])))
        img[y:y + ps, x:x + ps] = blend
    Image.fromarray(img).resize((OUT_SIZE, OUT_SIZE)).save(out_pdf, 'PDF', resolution=600.0)


def concept_legend(labels, out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(3, 2))
    fig.legend(handles=[Patch(facecolor=np.array(BRIGHT8[i]) / 255, edgecolor=None, label=l[:40])
                        for i, l in enumerate(labels)], loc='center', ncol=1, fontsize=8)
    fig.savefig(out_pdf, dpi=600, bbox_inches='tight', pad_inches=0); plt.close(fig)


def colorbar(out_pdf, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.5, 2.4))
    sm = plt.cm.ScalarMappable(cmap='jet', norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(['low', 'high'])
    cb.set_label(label, fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slide-id', required=True)
    ap.add_argument('--features-root', required=True, help='contains {fm}/{slide-id}.h5')
    ap.add_argument('--organ', default='auto', choices=['brain', 'lung', 'kidney', 'auto'],
                    help="'auto' scores the slide with all three organ models (use for slides of unknown or other organs)")
    ap.add_argument('--fms', nargs='+', default=VISION_FMS)
    ap.add_argument('--wsi', default=None, help='optional .svs drawn underneath the maps')
    ap.add_argument('--thumbnail', default=None, help='optional thumbnail (used when --wsi is not given)')
    ap.add_argument('--patch-px', type=int, default=None, help='tile size in level-0 pixels (read from h5 attrs if present)')
    ap.add_argument('--top-k', type=int, default=TOP_K, help='number of concepts mapped (max 8)')
    ap.add_argument('--run-id', type=int, default=0)
    ap.add_argument('--concept-root', default=os.path.join(ROOT, 'pretrained/concepts'))
    ap.add_argument('--weights-root', default=os.path.join(ROOT, 'pretrained/weights'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'demo/outputs'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    top_k = min(a.top_k, len(BRIGHT8))
    name = a.slide_id.split('.')[0]          # output prefix: TCGA barcode up to DX1 (drops the file UUID)

    tiles = {fm: load_tiles(os.path.join(a.features_root, fm, a.slide_id + '.h5')) for fm in a.fms}
    coords, attrs = tiles[a.fms[0]][1], tiles[a.fms[0]][2]
    for fm in a.fms[1:]:
        if tiles[fm][1] is not None and coords is not None and not np.array_equal(tiles[fm][1], coords):
            raise ValueError(f'{fm} tiles do not share the tile grid of {a.fms[0]}; extract all FMs from the same CLAM patches')
    patch_px = a.patch_px or int(attrs.get('patch_size', 512) * 2 ** int(attrs.get('patch_level', 0)))
    dims = (int(attrs['slide_width']), int(attrs['slide_height'])) if 'slide_width' in attrs else None
    if dims is None and coords is not None:     # no slide size: use the extent of the tile grid
        dims = (int(coords[:, 0].max() + patch_px), int(coords[:, 1].max() + patch_px))
    canvas = Canvas(a.wsi, a.thumbnail, dims) if coords is not None else None

    organs = list(ORGANS) if a.organ == 'auto' else [a.organ]
    report = {'slide_id': name, 'foundation_models': a.fms, 'organ_models': organs, 'per_organ': {}}
    tile_mi_organs = []                                             # tile-level MI per organ model
    for organ in organs:
        prev = ORGANS[organ]
        classes = list(LABEL_MAP[prev].keys())
        P, ATT, PC, PT = [], [], [], []
        names = None
        for fm in a.fms:
            feats = tiles[fm][0]
            m = load_model(prev, fm, feats.shape[1], dev, a.concept_root, a.weights_root, a.run_id)
            if names is None:
                names, keep = unique_concepts(m)
                owners = concept_owner(m, names, prev)
            p, att, pc, pt = run_fm(m, feats, dev, keep)
            P.append(p); ATT.append(att); PC.append(pc); PT.append(pt)
        P, ATT, PC, PT = map(np.stack, (P, ATT, PC, PT))
        pbar = P.mean(0)
        att = ATT.mean(0)                                           # mean of per-FM min-max attention

        # concept activation (plot_conceptlevel_map, use_attention=True): keep above-average-attention tiles
        sel = np.where(att >= att.mean())[0]
        pc = PC[:, sel].mean(0)                                     # [n_sel, U], averaged over FMs
        slide_concept = pc.mean(0)                                  # slide-level concept probability
        tile_concept = pc.argmax(1)
        top = [k for k, _ in Counter(tile_concept.tolist()).most_common(top_k)]
        top = sorted(top, key=lambda k: -slide_concept[k])          # order by slide-level probability
        remap = {k: i for i, k in enumerate(top)}
        lab = np.array([remap.get(k, -1) for k in tile_concept])
        mapped = lab >= 0

        unc = uncertainty(P)
        tile_mi_organs.append(entropy(PT.mean(0)) - entropy(PT).mean(0))   # tile-level MI across FMs
        report['per_organ'][organ] = {
            'predicted_subtype': classes[int(pbar.argmax())],
            'probabilities': {c: round(float(v), 4) for c, v in zip(classes, pbar)},
            'per_fm_probabilities': {fm: [round(float(v), 4) for v in P[i]] for i, fm in enumerate(a.fms)},
            'uncertainty': unc,
            'concept_tiles': int(len(sel)),
            'top_concepts': [{'concept': names[k], 'subtype': owners[k], 'probability': round(float(slide_concept[k]), 4)}
                             for k in top],
        }

        tag = f'{name}_{organ}'
        with open(os.path.join(a.out, tag + '_concepts.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['probability', 'subtype', 'concept'])
            for k in np.argsort(-slide_concept):
                w.writerow([f'{100 * slide_concept[k]:.4f}%', owners[k], names[k]])
        if canvas is not None:
            smooth_map(canvas, coords, att, patch_px, os.path.join(a.out, tag + '_attention.pdf'),
                       ATT_SIGMA, ATT_ALPHA, grey=GREY_ATT)
            concept_map(canvas, coords[sel][mapped], lab[mapped], patch_px, os.path.join(a.out, tag + '_concepts.pdf'))
            concept_legend([f'({100 * slide_concept[k]:.2f}%, {owners[k]}) {names[k].lower()}' for k in top],
                           os.path.join(a.out, tag + '_concepts_legend.pdf'))

    if canvas is not None:
        # one MI map per slide; outside the training scope, MI is averaged over the organ models
        # (as for the slide-level out_of_scope_scores)
        smooth_map(canvas, coords, np.mean(tile_mi_organs, 0), patch_px,
                   os.path.join(a.out, f'{name}_mutual_information.pdf'), UNC_SIGMA, UNC_ALPHA, grey=GREY_UNC)
        # colour scales are the same for every slide (maps are normalised per slide): one pair per folder
        colorbar(os.path.join(a.out, 'colorbar_attention.pdf'), 'Attention (normalised)')
        colorbar(os.path.join(a.out, 'colorbar_mutual_information.pdf'), 'Mutual information (normalised)')

    if len(organs) > 1:   # outside the training scope: average the three organ models' scores
        keys = ['normalized_entropy', 'msp', 'mutual_information']
        report['out_of_scope_scores'] = {k: float(np.mean([report['per_organ'][o]['uncertainty'][k] for o in organs])) for k in keys}
        report['note'] = ('Organ models were trained on malignant brain, lung or kidney tumours only. For slides outside '
                          'this scope the predicted subtype is not meaningful; high entropy and low MSP indicate the slide '
                          'should be deferred to a pathologist.')
    with open(os.path.join(a.out, f'{name}_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
