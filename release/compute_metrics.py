#!/usr/bin/env python3
"""
Recompute balanced accuracy (BACC) and macro one-vs-rest AUROC from the released prediction files,
with 95% CIs from patient-level bootstrap resampling (1,000 resamples by default).

    python release/compute_metrics.py --files cosmo_zenodo_data/predictions/**/*.csv \
        --slide-to-patient cosmo_zenodo_data/slide_to_patient/*.csv --out metrics.csv

Several prediction files of the same cohort and split (e.g. one per foundation model) can be
ensembled by averaging their softmax outputs with --ensemble, as done for the main results.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def load(path):
    d = pd.read_csv(path)
    return d['slide'].astype(str).values, d['label'].astype(int).values, d.iloc[:, 2:].values.astype(float), list(d.columns[2:])


def metrics(y, p):
    p = p / p.sum(1, keepdims=True)
    out = {'bacc': balanced_accuracy_score(y, p.argmax(1))}
    try:
        out['auroc'] = roc_auc_score(y, p[:, 1] if p.shape[1] == 2 else p, multi_class='ovr', average='macro',
                                     labels=None if p.shape[1] == 2 else list(range(p.shape[1])))
    except ValueError:
        out['auroc'] = np.nan
    return out


def bootstrap(slides, y, p, pat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series([pat.get(s, s) for s in slides])
    idx_by = groups.groupby(groups).indices
    keys = list(idx_by)
    res = []
    for _ in range(n):
        idx = np.concatenate([idx_by[k] for k in rng.choice(keys, len(keys), replace=True)])
        if len(np.unique(y[idx])) < 2: continue
        res.append(metrics(y[idx], p[idx]))
    r = pd.DataFrame(res)
    return {f'{m}_lo': np.nanpercentile(r[m], 2.5) for m in r} | {f'{m}_hi': np.nanpercentile(r[m], 97.5) for m in r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--slide-to-patient', nargs='*', default=[])
    ap.add_argument('--ensemble', action='store_true', help='average softmax over all --files (same slides)')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', default='metrics.csv')
    a = ap.parse_args()
    files = sorted({f for pat in a.files for f in glob.glob(pat, recursive=True)})
    pat = {}
    for f in a.slide_to_patient:
        for g in glob.glob(f):
            m = pd.read_csv(g); pat.update(dict(zip(m['slide'].astype(str), m['patient'].astype(str))))
    rows = []
    if a.ensemble:
        loaded = [load(f) for f in files]
        s0 = loaded[0][0]
        P = np.mean([pd.DataFrame(p, index=s).loc[s0].values for s, _, p, _ in loaded], 0)
        jobs = [('ensemble(%d files)' % len(files), s0, loaded[0][1], P)]
    else:
        jobs = [(f,) + load(f)[:3] for f in files]
    for name, s, y, p in jobs:
        r = {'file': name, 'n': len(y)} | metrics(y, p)
        if a.n_boot: r |= bootstrap(s, y, p, pat, a.n_boot)
        rows.append(r); print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
