"""
GO:BP enrichment via g:Profiler REST API.
Called from run_great.R; can also be run directly.
Uses TF gene names extracted from HOMER known motif results as the query set.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

GPROFILE_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

GENOME_TO_ORGANISM = {
    "hg38": "hsapiens", "hg19": "hsapiens",
    "mm10": "mmusculus", "mm9":  "mmusculus",
}

LAYERS = ["early", "mid", "late"]
PAIRS  = ["blood", "liver", "lymph"]
SIDES  = ["vitro", "vivo"]


def extract_tf_genes_from_homer(homer_result_txt: Path, n: int = 20) -> list[str]:
    """Extract TF gene names from HOMER knownResults.txt (top-n by p-value)."""
    if not homer_result_txt.exists():
        return []
    try:
        df = pd.read_csv(homer_result_txt, sep="\t", comment="#")
        pval_col = next(
            (c for c in df.columns if "p-value" in c.lower() or "pvalue" in c.lower()),
            None,
        )
        if pval_col is None:
            return []
        df = df.sort_values(pval_col).head(n)
        names = df.iloc[:, 0].str.split(r"[(/]").str[0].str.strip()
        # Keep only plausible gene names: 2-10 chars, letters/digits
        genes = [g for g in names if re.match(r"^[A-Za-z][A-Za-z0-9]{1,9}$", g)]
        return list(dict.fromkeys(genes))  # deduplicate, preserve order
    except Exception:
        return []


def gprofile_query(genes: list[str], organism: str) -> pd.DataFrame:
    """Query g:Profiler REST API for GO:BP enrichment."""
    payload = {
        "organism":       organism,
        "query":          genes,
        "sources":        ["GO:BP"],
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "no_evidences":   True,
    }
    try:
        resp = requests.post(GPROFILE_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", [])
        if not result:
            return pd.DataFrame()
        rows = []
        for r in result:
            rows.append({
                "term_id":           r.get("native", ""),
                "term_name":         r.get("name", ""),
                "p_value":           r.get("p_value", 1.0),
                "term_size":         r.get("term_size", 0),
                "query_size":        r.get("query_size", 0),
                "intersection_size": r.get("intersection_size", 0),
            })
        return pd.DataFrame(rows).sort_values("p_value")
    except Exception as e:
        print(f"    [WARN] g:Profiler API error: {e}")
        return pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top_features_dir", default="/workspace/outputs/top_features")
    p.add_argument("--annotation_dir",   default="/workspace/outputs/annotation/go")
    p.add_argument("--layers",           default="early mid late")
    p.add_argument("--n_top",            type=int, default=50)
    p.add_argument("--genome",           default="hg38")
    args = p.parse_args()

    annotation_dir = Path(args.annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    homer_dir = annotation_dir.parent / "homer"
    organism  = GENOME_TO_ORGANISM.get(args.genome, "hsapiens")
    layers    = args.layers.split()

    print(f"GO enrichment — organism={organism}, layers={layers}")

    for layer in layers:
        for side in SIDES:
            for pair in PAIRS:
                out_tsv = annotation_dir / f"{layer}_{side}_{pair}_go.tsv"
                if out_tsv.exists():
                    print(f"  [SKIP] {layer}/{side}/{pair}")
                    continue

                homer_txt = homer_dir / f"{layer}_{side}_{pair}" / "knownResults.txt"
                genes = extract_tf_genes_from_homer(homer_txt)
                print(f"  {layer}/{side}/{pair}: {len(genes)} TF genes from HOMER")

                if len(genes) < 3:
                    print(f"    Too few genes — writing empty TSV")
                    pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
                    continue

                print(f"    Querying g:Profiler: {genes[:5]}...")
                df = gprofile_query(genes, organism)
                df.to_csv(out_tsv, sep="\t", index=False)
                if not df.empty:
                    print(f"    {len(df)} GO:BP terms saved → {out_tsv}")
                else:
                    print(f"    No significant terms")

    print("GO enrichment done.")


if __name__ == "__main__":
    main()
