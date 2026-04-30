#!/usr/bin/env python3
"""
Generate per-feature BED files from SAE activations for HOMER motif analysis.

For each layer × pair × side combination, identifies the top-50 context-divergent
SAE features (by CDS score for that pair) and writes a BED file of each feature's
top-200 most-activated windows in the relevant condition.

Output layout expected by run_annotation_v2.sh:
    outputs/top_features/{layer}/feature_{id}/top_windows_{side}_{pair}.bed
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / 'src'))
from sae import BatchTopKSAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

LAYERS = ['early', 'mid', 'late']
PAIRS  = ['blood', 'liver', 'lymph']
SIDES  = ['vitro', 'vivo']

PAIR_CONDS = {
    'blood': {'vitro': 'K562',    'vivo': 'HSC'},
    'liver': {'vitro': 'HepG2',   'vivo': 'Liver'},
    'lymph': {'vitro': 'GM12878', 'vivo': 'NaiveB'},
}

N_TOP_FEATURES = 50   # top CDS features per layer/pair/side
N_TOP_WINDOWS  = 200  # top-activated windows per feature

OUT_BASE = REPO / 'outputs' / 'top_features'


def load_windows():
    windows = []
    with open(REPO / 'data' / 'windows.bed') as f:
        for line in f:
            parts = line.strip().split('\t')
            windows.append((parts[0], int(parts[1]), int(parts[2])))
    return windows


def encode_sae(sae, acts, batch_size=2048):
    n = len(acts)
    z = np.zeros((n, sae.d_latent), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(acts[start:end]).float().to(DEVICE)
        with torch.no_grad():
            z[start:end] = sae.encode(x).cpu().numpy()
    return z


def main():
    windows = load_windows()
    print(f"Loaded {len(windows)} windows from windows.bed")

    for layer in LAYERS:
        print(f"\n{'='*60}\nLayer: {layer}")

        sae = BatchTopKSAE.load(str(REPO / 'saes' / layer / 'pooled.pt'), device=DEVICE)
        sae.eval()
        print(f"  SAE: d_latent={sae.d_latent}")

        cds_df = pd.read_csv(REPO / 'results' / 'cds' / f'layer_{layer}_features.tsv', sep='\t')

        # Encode all 6 conditions once and cache
        encoded = {}
        for pair in PAIRS:
            for side in SIDES:
                cond = PAIR_CONDS[pair][side]
                acts = torch.load(
                    REPO / 'activations' / pair / cond / f'{layer}.pt',
                    map_location='cpu', weights_only=True
                ).numpy()
                print(f"  Encoding {pair}/{side} ({cond})...", end=' ', flush=True)
                encoded[(pair, side)] = encode_sae(sae, acts)
                print(f"done.")

        # Write BED files for each pair/side
        for pair in PAIRS:
            for side in SIDES:
                cds_col = f'cds_{pair}'

                if side == 'vitro':
                    top_feats = (cds_df[cds_df[cds_col] > 0]
                                 .nlargest(N_TOP_FEATURES, cds_col))
                else:
                    top_feats = (cds_df[cds_df[cds_col] < 0]
                                 .nsmallest(N_TOP_FEATURES, cds_col))

                if top_feats.empty:
                    print(f"  WARNING: no features for {layer}/{side}/{pair}")
                    continue

                z = encoded[(pair, side)]
                n_written = 0

                for _, row in top_feats.iterrows():
                    feat_id   = int(row['feature_id'])
                    top_idx   = np.argsort(z[:, feat_id])[::-1][:N_TOP_WINDOWS]

                    out_dir  = OUT_BASE / layer / f'feature_{feat_id:05d}'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    bed_path = out_dir / f'top_windows_{side}_{pair}.bed'

                    with open(bed_path, 'w') as f:
                        for idx in top_idx:
                            chrom, start, end = windows[idx]
                            f.write(f'{chrom}\t{start}\t{end}\n')

                    n_written += 1

                print(f"  {layer}/{side}/{pair}: {n_written} BED files "
                      f"({N_TOP_FEATURES} features × {N_TOP_WINDOWS} windows)")

    print(f"\nDone. BED files written to: {OUT_BASE}")


if __name__ == '__main__':
    main()
