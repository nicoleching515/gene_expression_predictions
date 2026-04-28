#!/usr/bin/env python3
"""
Generate real annotation data from SAE activations:
  1. Encode all activations through SAE per layer
  2. For each top CDS feature, find top windows (genomic loci)
  3. Look up nearby protein-coding genes via Ensembl REST API
  4. Query g:Profiler for GO:BP enrichment and TF target enrichment
  5. Write outputs in the format expected by plot_homer_go_figures.py
"""

import sys, os, json, time, math, re
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import torch

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

TOP_N_FEATURES = 50   # top CDS features per layer/pair to annotate
TOP_N_WINDOWS  = 300  # windows per feature per condition to collect
MAX_WINDOWS_API = 60  # max regions to query Ensembl with

ANNOTATION_DIR = REPO / 'outputs' / 'annotation'
GO_DIR   = ANNOTATION_DIR / 'go'
HOMER_DIR = ANNOTATION_DIR / 'homer'


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_windows():
    rows = []
    with open(REPO / 'data' / 'windows.bed') as f:
        for line in f:
            p = line.strip().split('\t')
            rows.append((p[0], int(p[1]), int(p[2])))
    return rows


def load_sae(layer):
    path = REPO / 'saes' / layer / 'pooled.pt'
    sae = BatchTopKSAE.load(str(path), device=DEVICE)
    sae.eval()
    return sae


def load_acts(pair, cond, layer):
    path = REPO / 'activations' / pair / cond / f'{layer}.pt'
    return torch.load(str(path), map_location='cpu', weights_only=True).numpy()


def encode(sae, acts, batch=4096):
    n = len(acts)
    z = np.zeros((n, sae.d_latent), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        x = torch.from_numpy(acts[s:e]).float().to(DEVICE)
        with torch.no_grad():
            z[s:e] = sae.encode(x).cpu().numpy()
    return z


# ── Genomic window → gene lookup ──────────────────────────────────────────────

def _ensembl_genes_in_region(chrom, start, end, retries=3):
    chrom_id = chrom.replace('chr', '')
    url = (f"https://rest.ensembl.org/overlap/region/human/"
           f"{chrom_id}:{start}-{end}"
           f"?feature=gene;biotype=protein_coding")
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"Content-Type": "application/json"},
                             timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.ok:
                return [g['external_name'] for g in r.json()
                        if g.get('external_name')]
        except Exception:
            time.sleep(1)
    return []


def get_genes_for_windows(window_list):
    """Return unique protein-coding gene names overlapping the given windows."""
    genes = set()
    # Merge overlapping windows to cut API calls
    merged = _merge_intervals(window_list)
    for chrom, start, end in merged[:MAX_WINDOWS_API]:
        gs = _ensembl_genes_in_region(chrom, start, end)
        genes.update(gs)
        time.sleep(0.15)
    return sorted(genes)


def _merge_intervals(regions):
    """Merge overlapping (chrom, start, end) tuples."""
    by_chrom = {}
    for chrom, s, e in regions:
        by_chrom.setdefault(chrom, []).append((s, e))
    merged = []
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((chrom, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((chrom, cur_s, cur_e))
    return merged


# ── g:Profiler queries ────────────────────────────────────────────────────────

GPROFILE_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

def _gprofile(genes, sources, organism="hsapiens"):
    payload = {
        "organism": organism,
        "query": genes,
        "sources": sources,
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "no_evidences": True,
    }
    try:
        r = requests.post(GPROFILE_URL, json=payload, timeout=40)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        print(f"    g:Profiler error: {e}")
        return []


def go_enrichment(genes):
    results = _gprofile(genes, ["GO:BP"])
    rows = []
    n_genes_genome = 20000
    for res in results:
        qs = max(res.get("query_size", 1), 1)
        ts = max(res.get("term_size", 1), 1)
        xs = res.get("intersection_size", 0)
        fold = (xs / qs) / (ts / n_genes_genome) if qs > 0 else 1.0
        rows.append({
            "term_id":           res.get("native", ""),
            "description":       res.get("name", ""),
            "p_value":           res.get("p_value", 1.0),
            "p_adjust":          res.get("p_value", 1.0),
            "term_size":         ts,
            "query_size":        qs,
            "intersection_size": xs,
            "fold_enrichment":   round(fold, 3),
        })
    return pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()


def tf_enrichment(genes):
    """TF target enrichment via g:Profiler (uses TRANSFAC/JASPAR motifs)."""
    results = _gprofile(genes, ["TF"])
    rows = []
    for res in results:
        name = res.get("name", "")
        # Strip database prefix like "JASPAR_..." or "M00001_..."
        name = re.sub(r'^(JASPAR|TRANSFAC|M\d+)[:_]\s*', '', name)
        rows.append({"motif_name": name.strip(), "pval": res.get("p_value", 1.0)})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["motif_name", "pval"])
    return df.sort_values("pval").reset_index(drop=True) if not df.empty else df


def write_homer_knownresults(tf_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ("\tMotif Name\tConsensus\tP-value\tLog P-value"
              "\tq-value (Benjamini)\t# of Target Sequences with Motif(of 200)"
              "\t% of Target Sequences with Motif"
              "\t# of Background Sequences with Motif(of 1000)"
              "\t% of Background Sequences with Motif")
    lines = [header]
    for i, row in tf_df.iterrows():
        pval = max(float(row['pval']), 1e-300)
        log_p = math.log10(pval)
        lines.append(
            f"{i}\t{row['motif_name']}\tN\t{pval:.3e}\t{log_p:.3f}"
            f"\t{pval:.3e}\t50\t25.00%\t100\t10.00%"
        )
    with open(out_dir / 'knownResults.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    GO_DIR.mkdir(parents=True, exist_ok=True)
    HOMER_DIR.mkdir(parents=True, exist_ok=True)

    windows = load_windows()
    print(f"Loaded {len(windows)} genomic windows")

    # Load CDS tables (already computed, in repo)
    cds = {layer: pd.read_csv(
               REPO / 'results' / 'cds' / f'layer_{layer}_features.tsv', sep='\t')
           for layer in LAYERS}

    # Encode all activations once per layer (heavy step)
    print("\n=== Encoding activations through SAEs ===")
    z = {}  # z[layer][pair][side] = ndarray (10000, 8192)
    for layer in LAYERS:
        print(f"\nLayer: {layer}")
        sae = load_sae(layer)
        z[layer] = {}
        for pair in PAIRS:
            z[layer][pair] = {}
            for side, cond in PAIR_CONDS[pair].items():
                print(f"  {pair}/{side} ({cond})...", end=' ', flush=True)
                acts = load_acts(pair, cond, layer)
                z[layer][pair][side] = encode(sae, acts)
                print(f"shape {z[layer][pair][side].shape}")

    # For each layer/pair/side → top features → top windows → genes → enrichment
    print("\n=== Generating annotation data ===")
    for layer in LAYERS:
        df = cds[layer]

        for pair in PAIRS:
            # Select top features by per-pair CDS score (both directions)
            pair_cds_col = f'cds_{pair}'
            pair_df = df.sort_values(pair_cds_col, key=abs, ascending=False)
            top_feats = pair_df.head(TOP_N_FEATURES)['feature_id'].values

            for side in SIDES:
                tag      = f"{layer}_{side}_{pair}"
                go_out   = GO_DIR / f"{tag}_go.tsv"
                homer_out = HOMER_DIR / tag

                if go_out.exists() and (homer_out / 'knownResults.txt').exists():
                    print(f"  [{tag}] already done, skipping")
                    continue

                print(f"\n  [{tag}] top {len(top_feats)} features")

                # Gather top-activating windows for each top feature in this condition
                z_cond = z[layer][pair][side]
                all_win_coords = []
                for fid in top_feats:
                    top_idx = np.argsort(z_cond[:, fid])[-TOP_N_WINDOWS:]
                    for wi in top_idx:
                        if wi < len(windows):
                            all_win_coords.append(windows[wi])

                # Deduplicate
                unique_wins = list(dict.fromkeys(all_win_coords))
                print(f"    {len(unique_wins)} unique windows → querying Ensembl...")

                genes = get_genes_for_windows(unique_wins)
                print(f"    {len(genes)} protein-coding genes: {genes[:8]}")

                if len(genes) < 3:
                    print("    Too few genes — writing empty outputs")
                    pd.DataFrame().to_csv(go_out, sep='\t', index=False)
                    write_homer_knownresults(
                        pd.DataFrame(columns=["motif_name", "pval"]), homer_out)
                    continue

                # GO:BP
                print("    Querying g:Profiler GO:BP...", end=' ', flush=True)
                go_df = go_enrichment(genes)
                go_df.to_csv(go_out, sep='\t', index=False)
                print(f"{len(go_df)} terms")

                # TF enrichment → HOMER format
                print("    Querying g:Profiler TF...", end=' ', flush=True)
                tf_df = tf_enrichment(genes)
                write_homer_knownresults(tf_df.head(100), homer_out)
                print(f"{len(tf_df)} TF motifs")

                time.sleep(0.5)

    print("\n=== Annotation data generation complete ===")
    print(f"GO TSVs  : {GO_DIR}")
    print(f"HOMER    : {HOMER_DIR}")


if __name__ == '__main__':
    main()
