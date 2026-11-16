#!/usr/bin/env python3
"""
COSMO demo: slide prediction, attention heatmap, concept activation map and uncertainty.

Runs the released COSMO adapters on pre-extracted tile features of one slide, for one
organ model (brain, lung or kidney) or, for a slide outside the training scope, for all
three organ models. Mirrors the analyses in the paper:

  * prediction  : mean-softmax ensemble over foundation models (FMs)
  * attention   : gated-attention weight of each tile (per FM, averaged over FMs)
  * concepts    : P_j(x) = softmax_j( tau * cos(x, c_j) ) for each tile embedding x,
                  averaged over high-attention tiles; the map colours each of these
                  tiles by its highest-scoring concept (STAR Methods, concept mapping)
  * uncertainty : normalised entropy H(p)/log C, maximum softmax probability (MSP) and
                  mutual information (MI) across FMs; for slides outside the training
                  scope the three organ models are averaged (STAR Methods, OOD analysis)
  * MI map      : tile-level mutual information across FMs (Figure S7f)

Concept scores are relative shares that sum to one over the organ's concepts. They are
model-derived associations, not the probability that a feature is present.

Inputs (per FM): a CLAM-style .h5 file with datasets `features` [N, D] and `coords` [N, 2]
(level-0 pixel coordinates of each tile). The .svs file is optional and only used to draw
the thumbnail underneath the maps.

Example:
    python demo/cosmo_demo.py --slide-id TCGA-XX-XXXX-01Z-00-DX1 \
        --features-root ./demo/data/features --wsi ./demo/data/slides/TCGA-XX.svs \
        --organ lung --fms uni gigapath chief --out ./demo/outputs

    # slide from an organ outside the training scope (e.g. breast): score with all organs
    python demo/cosmo_demo.py --slide-id TCGA-BRCA-... --organ auto ...
"""
import argparse, json, os, sys

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


@torch.no_grad()
def run_fm(model, feats, device):
    x = feats.to(device)
    slide_logits = model.inference(x.unsqueeze(0))[0]                 # [C]
    p_slide = F.softmax(slide_logits, -1).cpu().numpy()
    h = model.visual_projection(x)                                      # tile embeddings [N, 512]
    att = F.softmax(model.attention(h)[:, 0], 0).cpu().numpy()          # [N]
    tau = model.logit_scale.exp()
    cos = F.normalize(h, dim=-1) @ F.normalize(model.concepts.to(device), dim=-1).t()
    p_concept = F.softmax(tau * cos, -1).cpu().numpy()                  # [N, n_concepts]
    model.mode = 1                                                      # tile-level class scores
    tile_logits = model.inference(x)
    model.mode = 0
    p_tile = F.softmax(tile_logits, -1).cpu().numpy()                   # [N, C]
    return p_slide, att, p_concept, p_tile


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis)


def uncertainty(p_fms):
    """p_fms: [F, C] slide-level softmax per FM."""
    pbar = p_fms.mean(0)
    C = pbar.shape[-1]
    H = entropy(pbar)
    return {'normalized_entropy': float(H / np.log(C)), 'msp': float(pbar.max()),
            'mutual_information': float(H - entropy(p_fms).mean())}


def draw_maps(out_png, coords, patch_px, values, wsi, title, cmap='jet', categorical=None, legend=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    W = coords[:, 0].max() + patch_px
    H = coords[:, 1].max() + patch_px
    scale = 1.0
    fig, ax = plt.subplots(figsize=(8, 8 * H / W))
    if wsi is not None:
        try:
            import openslide
            s = openslide.OpenSlide(wsi)
            thumb = s.get_thumbnail((2000, 2000))
            scale = thumb.size[0] / s.dimensions[0]
            ax.imshow(thumb)
        except Exception as e:  # thumbnail is optional
            print('thumbnail skipped:', e)
    sz = patch_px * scale
    if categorical is not None:
        palette = plt.get_cmap('tab10')
        for (cx, cy), k in zip(coords, categorical):
            if k < 0:
                continue
            ax.add_patch(plt.Rectangle((cx * scale, cy * scale), sz, sz, color=palette(k % 10), alpha=0.55, lw=0))
        if legend:
            ax.legend(handles=[Patch(color=palette(i % 10), label=l) for i, l in enumerate(legend)],
                      loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    else:
        v = (values - values.min()) / (np.ptp(values) + 1e-12)
        cm = plt.get_cmap(cmap)
        for (cx, cy), a in zip(coords, v):
            ax.add_patch(plt.Rectangle((cx * scale, cy * scale), sz, sz, color=cm(a), alpha=0.5, lw=0))
    if wsi is None:
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_axis_off(); ax.set_title(title, fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slide-id', required=True)
    ap.add_argument('--features-root', required=True, help='contains {fm}/{slide-id}.h5')
    ap.add_argument('--organ', default='auto', choices=['brain', 'lung', 'kidney', 'auto'],
                    help="'auto' scores the slide with all three organ models (use for slides of unknown or other organs)")
    ap.add_argument('--fms', nargs='+', default=VISION_FMS)
    ap.add_argument('--wsi', default=None, help='optional .svs for the thumbnail')
    ap.add_argument('--patch-px', type=int, default=None, help='tile size in level-0 pixels (read from h5 attrs if present)')
    ap.add_argument('--top-frac', type=float, default=0.1, help='fraction of highest-attention tiles used for concepts')
    ap.add_argument('--top-k', type=int, default=8, help='number of concepts reported and mapped')
    ap.add_argument('--run-id', type=int, default=0)
    ap.add_argument('--concept-root', default=os.path.join(ROOT, 'pretrained/concepts'))
    ap.add_argument('--weights-root', default=os.path.join(ROOT, 'pretrained/weights'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'demo/outputs'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)

    tiles = {fm: load_tiles(os.path.join(a.features_root, fm, a.slide_id + '.h5')) for fm in a.fms}
    coords, attrs = tiles[a.fms[0]][1], tiles[a.fms[0]][2]
    for fm in a.fms[1:]:
        if tiles[fm][1] is not None and coords is not None and not np.array_equal(tiles[fm][1], coords):
            raise ValueError(f'{fm} tiles do not share the tile grid of {a.fms[0]}; extract all FMs from the same CLAM patches')
    patch_px = a.patch_px or int(attrs.get('patch_size', 512) * 2 ** int(attrs.get('patch_level', 0)))

    organs = list(ORGANS) if a.organ == 'auto' else [a.organ]
    report = {'slide_id': a.slide_id, 'foundation_models': a.fms, 'organ_models': organs, 'per_organ': {}}
    for organ in organs:
        prev = ORGANS[organ]
        classes = list(LABEL_MAP[prev].keys())
        P, ATT, PC, PT = [], [], [], []
        for fm in a.fms:
            feats = tiles[fm][0]
            m = load_model(prev, fm, feats.shape[1], dev, a.concept_root, a.weights_root, a.run_id)
            concept_list = m.concept_list
            p, att, pc, pt = run_fm(m, feats, dev)
            P.append(p); ATT.append(att); PC.append(pc); PT.append(pt)
        P, ATT, PC, PT = map(np.stack, (P, ATT, PC, PT))
        pbar = P.mean(0)
        att = ATT.mean(0)
        top = np.argsort(-att)[:max(1, int(len(att) * a.top_frac))]
        concept_scores = PC[:, top].mean((0, 1))                          # relative share over concepts
        order = np.argsort(-concept_scores)[:a.top_k]
        unc = uncertainty(P)
        # tile-level MI across FMs
        tbar = PT.mean(0)
        tile_mi = entropy(tbar) - entropy(PT).mean(0)
        report['per_organ'][organ] = {
            'predicted_subtype': classes[int(pbar.argmax())],
            'probabilities': {c: round(float(v), 4) for c, v in zip(classes, pbar)},
            'per_fm_probabilities': {fm: [round(float(v), 4) for v in P[i]] for i, fm in enumerate(a.fms)},
            'uncertainty': unc,
            'top_concepts': [{'concept': concept_list[j], 'share': round(float(concept_scores[j]), 4)} for j in order],
        }
        if coords is not None:
            tag = f'{a.slide_id}_{organ}'
            draw_maps(os.path.join(a.out, tag + '_attention.png'), coords, patch_px, att, a.wsi,
                      f'{organ} model: attention (mean over {len(a.fms)} FMs)')
            best = PC[:, :, order].mean(0).argmax(-1)                       # top-k concept per tile
            cat = np.full(len(att), -1); cat[top] = best[top]
            draw_maps(os.path.join(a.out, tag + '_concepts.png'), coords, patch_px, None, a.wsi,
                      f'{organ} model: concept activation map (top {len(top)} attention tiles)',
                      categorical=cat, legend=[concept_list[j] for j in order])
            draw_maps(os.path.join(a.out, tag + '_mutual_information.png'), coords, patch_px, tile_mi, a.wsi,
                      f'{organ} model: tile-level mutual information across FMs', cmap='magma')

    if len(organs) > 1:   # outside the training scope: average the three organ models' scores
        keys = ['normalized_entropy', 'msp', 'mutual_information']
        report['out_of_scope_scores'] = {k: float(np.mean([report['per_organ'][o]['uncertainty'][k] for o in organs])) for k in keys}
        report['note'] = ('Organ models were trained on malignant brain, lung or kidney tumours only. For slides outside '
                          'this scope the predicted subtype is not meaningful; high entropy and low MSP indicate the slide '
                          'should be deferred to a pathologist.')
    with open(os.path.join(a.out, f'{a.slide_id}_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
