"""
Write PHASE_8_REPORT.md summarising HOMER, ChromHMM, and GO enrichment results.
"""

import argparse
import os
from pathlib import Path
import pandas as pd


CHROMHMM_STATE_NAMES = {
    '1': 'TssA', '2': 'TssAFlnk', '3': 'TxFlnk', '4': 'Tx',
    '5': 'TxWk', '6': 'EnhG', '7': 'Enh', '8': 'ZNF/Rpts',
    '9': 'Het', '10': 'TssBiv', '11': 'BivFlnk', '12': 'EnhBiv',
    '13': 'ReprPC', '14': 'ReprPCWk', '15': 'Quies',
}

LAYERS = ['early', 'mid', 'late']
PAIRS  = ['blood', 'liver', 'lymph']
SIDES  = ['vivo', 'vitro']


def top_motifs(homer_dir: Path, layer, side, pair, n=3):
    p = homer_dir / f'{layer}_{side}_{pair}' / 'homerResults' / 'knownResults.txt'
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, sep='\t', comment='#')
        pval_col = next((c for c in df.columns if 'p-value' in c.lower() or 'pvalue' in c.lower()), None)
        if pval_col is None:
            return []
        name_col = df.columns[0]
        return list(df.sort_values(pval_col).head(n)[name_col].str.split('(').str[0].str.strip())
    except Exception:
        return []


def dominant_state(chromhmm_dir: Path, layer, side, pair):
    p = chromhmm_dir / f'{layer}_{side}_{pair}_states.tsv'
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p, sep='\t')
        if df.empty:
            return None, None
        df['state_name'] = df['state'].astype(str).map(CHROMHMM_STATE_NAMES).fillna(df['state'].astype(str))
        top = df.sort_values('fraction', ascending=False).iloc[0]
        return top['state_name'], float(top['fraction'])
    except Exception:
        return None, None


def top_go_terms(go_dir: Path, layer, side, pair, n=3):
    p = go_dir / f'{layer}_{side}_{pair}_go.tsv'
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, sep='\t')
        if df.empty:
            return []
        pval_col = next((c for c in df.columns if 'adj' in c.lower() or 'fdr' in c.lower()), df.columns[-1])
        name_col = next((c for c in df.columns if 'name' in c.lower() or 'term' in c.lower()), df.columns[0])
        terms = list(df.sort_values(pval_col).head(n)[name_col])
        return [str(t)[:60] for t in terms]
    except Exception:
        return []


def write_report(annotation_dir: Path, output: Path):
    homer_dir    = annotation_dir / 'homer'
    chromhmm_dir = annotation_dir / 'chromhmm'
    go_dir       = annotation_dir / 'go'

    lines = [
        '# PHASE 8 REPORT — Bio Annotation of Top SAE Features',
        '',
        'HOMER motif enrichment, ChromHMM state annotation, and GO:BP enrichment',
        'for the top-50 context-divergent SAE features per layer.',
        '',
        '---',
        '',
    ]

    for layer in LAYERS:
        lines += [f'## Layer: {layer.upper()}', '']

        # ── HOMER ────────────────────────────────────────────────────────────
        lines += ['### Top HOMER Motifs', '',
                  '| Pair | Side | Top motifs |',
                  '|------|------|------------|']
        homer_any = False
        for pair in PAIRS:
            for side in SIDES:
                motifs = top_motifs(homer_dir, layer, side, pair)
                if motifs:
                    homer_any = True
                    lines.append(f'| {pair} | {side} | {", ".join(motifs)} |')
        if not homer_any:
            lines.append('*No HOMER results found — run Step 2.*')
        lines.append('')

        # ── ChromHMM ─────────────────────────────────────────────────────────
        lines += ['### Dominant ChromHMM State', '',
                  '| Pair | Side | State | Fraction |',
                  '|------|------|-------|----------|']
        chmm_any = False
        for pair in PAIRS:
            for side in SIDES:
                state, frac = dominant_state(chromhmm_dir, layer, side, pair)
                if state:
                    chmm_any = True
                    lines.append(f'| {pair} | {side} | {state} | {frac:.3f} |')
        if not chmm_any:
            lines.append('*No ChromHMM results found — run Step 3.*')
        lines.append('')

        # ── GO ───────────────────────────────────────────────────────────────
        lines += ['### Top GO:BP Terms', '',
                  '| Pair | Side | Top GO terms |',
                  '|------|------|--------------|']
        go_any = False
        for pair in PAIRS:
            for side in SIDES:
                terms = top_go_terms(go_dir, layer, side, pair)
                if terms:
                    go_any = True
                    lines.append(f'| {pair} | {side} | {"; ".join(terms)} |')
        if not go_any:
            lines.append('*No GO results found — run Step 4.*')
        lines += ['', '---', '']

    lines += [
        '## Figures',
        '',
        '| Figure | Path | Description |',
        '|--------|------|-------------|',
        '| Fig 3 | `results/figures/fig3_annotation_heatmap.{pdf,png}` | ChromHMM state enrichment heatmap |',
        '| Fig 6 | `results/figures/fig6_case_studies.{pdf,png}` | Motif + ChromHMM case studies |',
        '',
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotation_dir', default='/workspace/outputs/annotation')
    p.add_argument('--output',         default='/workspace/PHASE_8_REPORT.md')
    args = p.parse_args()
    write_report(Path(args.annotation_dir), Path(args.output))


if __name__ == '__main__':
    main()
