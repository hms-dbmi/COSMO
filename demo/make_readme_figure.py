#!/usr/bin/env python3
"""Compose docs/demo_examples.png (README figure) from the demo outputs.

Rows: one in-scope kidney slide, one in-scope lung slide, one breast slide outside the training scope.
Columns: slide thumbnail, attention, concept map, top concepts (legend), uncertainty (mutual information).
Run the demo with --mi-vmax 0.7 first so that the uncertainty maps share one scale.

    python demo/make_readme_figure.py --outputs <demo outputs> --thumbnails <thumbnails> --out docs/demo_examples.png
"""
import argparse, json, os, subprocess, tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROWS = [  # slide, organ model used for attention and concepts, row label
    ('TCGA-B9-4617-01Z-00-DX1', 'kidney', 'Kidney (in scope)'),
    ('TCGA-64-1680-01Z-00-DX1', 'lung', 'Lung (in scope)'),
    ('TCGA-A1-A0SD-01Z-00-DX1', 'lung', 'Breast (out of scope)'),
]
COLS = ['Slide', 'Attention', 'Concept map', 'Top concepts', 'Uncertainty (MI)']
ABBR = {'Papillary Renal Cell Carcinoma': 'PRCC', 'Lung Adenocarcinoma': 'LUAD', 'Lung Squamous Cell Carcinoma': 'LUSC'}


def pdf_png(pdf, dpi=150):
    tmp = tempfile.mkdtemp()
    subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', pdf, os.path.join(tmp, 'x')], check=True)
    return Image.open(os.path.join(tmp, 'x.png')).convert('RGB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--thumbnails', required=True)
    ap.add_argument('--mi-vmax', type=float, default=0.7)
    ap.add_argument('--out', default='docs/demo_examples.png')
    a = ap.parse_args()

    fig = plt.figure(figsize=(13.8, 7.6))
    gs = fig.add_gridspec(len(ROWS) + 1, 5, width_ratios=[1, 1, 1, 1.25, 1], height_ratios=[1] * len(ROWS) + [0.06],
                          left=0.15, right=0.995, top=0.94, bottom=0.04, wspace=0.04, hspace=0.05)
    for i, (sid, organ, label) in enumerate(ROWS):
        rep = json.load(open(os.path.join(a.outputs, f'{sid}_report.json')))
        po = rep['per_organ'][organ]
        pred = po['predicted_subtype']; p = max(po['probabilities'].values())
        u = rep.get('out_of_scope_scores', po['uncertainty'])
        sub = (f"{sid[:12]}\npredicted {ABBR.get(pred, pred)} ({p:.2f})"
               + (f"\n{organ} model" if 'out_of_scope_scores' in rep else '')
               + f"\nentropy {u['normalized_entropy']:.2f}, MI {u['mutual_information']:.2f}")
        imgs = [Image.open(os.path.join(a.thumbnails, sid + '.png')).convert('RGB').resize((512, 512)),
                pdf_png(os.path.join(a.outputs, f'{sid}_{organ}_attention.pdf')),
                pdf_png(os.path.join(a.outputs, f'{sid}_{organ}_concepts.pdf')),
                pdf_png(os.path.join(a.outputs, f'{sid}_{organ}_concepts_legend.pdf'), dpi=220),
                pdf_png(os.path.join(a.outputs, f'{sid}_mutual_information.pdf'))]
        for j, im in enumerate(imgs):
            ax = fig.add_subplot(gs[i, j]); ax.axis('off')
            ax.imshow(np.asarray(im))
            if i == 0:   # column headers at a common height
                bb = gs[0, j].get_position(fig)
                fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.012, COLS[j], ha='center', va='bottom', fontsize=12,
                         fontweight='bold')
            if j == 0:   # row label: organ (bold) and slide details
                ax.text(-0.06, 0.62, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5,
                        fontweight='bold')
                ax.text(-0.06, 0.6, sub, transform=ax.transAxes, ha='right', va='top', fontsize=9, linespacing=1.35)
    # colour scales under the attention and uncertainty columns
    for j, (lab, ticks) in {1: ('low  →  high attention', None), 4: (f'0 – {a.mi_vmax:g} nats', None)}.items():
        cax = fig.add_subplot(gs[len(ROWS), j])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=matplotlib.colors.Normalize(0, 1)), cax=cax,
                          orientation='horizontal', ticks=[])
        cb.set_label(lab, fontsize=9, labelpad=2); cb.outline.set_linewidth(0.5)
    fig.savefig(a.out, dpi=110)
    print('saved', a.out)


if __name__ == '__main__':
    main()
