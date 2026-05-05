#!/usr/bin/env python3
"""
src/parse_homer_denovo.py
=========================
Parser for HOMER *de novo* motif enrichment results (homerResults.html).

HOMER's findMotifsGenome.pl writes two result sets:
  - knownResults.txt   → enrichment of a pre-compiled TF database (already parsed)
  - homerResults.html  → motifs discovered *de novo* from your sequences, annotated
                         with the best known TF match from the database

This module parses the de novo HTML table and produces the same
DataFrame schema used by parse_homer_known() so the figures code works
with both result types transparently.

De novo HTML table columns (HOMER v4/v5):
  Rank | Motif logo | Name | Log Odds | P-value | log(P-value)
  % Target | % Bg | STD | Best Match / Details

Usage:
    from parse_homer_denovo import parse_homer_denovo, parse_homer_auto

    # de novo only
    df = parse_homer_denovo("/path/to/homer_dir", n=15)

    # auto: prefer de novo, fall back to known
    df = parse_homer_auto("/path/to/homer_dir", n=15, prefer="denovo")
"""

import re
import warnings
from pathlib import Path
from html.parser import HTMLParser

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Low-level HTML table extractor
# ─────────────────────────────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    """
    Extract all <table> rows from an HTML file.
    Returns list of rows, each row is a list of cell text strings.
    """

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False
        self._depth = 0  # nested table guard

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._current_table = []
        elif tag in ("tr",):
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            if self._depth == 1:
                self.tables.append(self._current_table)
                self._current_table = []
            self._depth -= 1
        elif tag == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(" ".join(self._current_cell).strip())
            self._current_cell = []

    def handle_data(self, data):
        if self._in_cell:
            stripped = data.strip()
            if stripped:
                self._current_cell.append(stripped)


def _extract_tables_from_html(html_path: Path) -> list[list[list[str]]]:
    """Parse HTML and return all table contents."""
    text = html_path.read_text(errors="replace")
    parser = _TableParser()
    parser.feed(text)
    return parser.tables


# ─────────────────────────────────────────────────────────────────────────────
# De novo result parser
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_DF = pd.DataFrame(
    columns=["motif_name", "pval", "qval", "neg_log_p", "fold_enrichment",
             "pct_target", "pct_bg", "best_known_match", "source"]
)

_PVAL_RE   = re.compile(r"([0-9]+(?:\.[0-9]+)?)[eE]([+-]?[0-9]+)")
_PCT_RE    = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_BEST_RE   = re.compile(r"Best Match:\s*(.+?)(?:\||$)", re.IGNORECASE)


def _parse_pval(cell: str) -> float:
    """Parse a p-value string like '1e-12' or '3.14e-5'."""
    cell = cell.strip()
    try:
        return float(cell)
    except ValueError:
        pass
    m = _PVAL_RE.search(cell)
    if m:
        try:
            return float(f"{m.group(1)}e{m.group(2)}")
        except ValueError:
            pass
    return np.nan


def _parse_pct(cell: str) -> float:
    """Extract first percentage number from a cell string."""
    m = _PCT_RE.search(cell)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return np.nan


def _clean_motif_name(raw: str) -> str:
    """
    Extract TF name from the 'Best Match' cell.
    Falls back to the de novo rank label if no TF match found.
    """
    # Try explicit "Best Match:" label
    m = _BEST_RE.search(raw)
    if m:
        name = m.group(1).strip()
    else:
        # Take everything before the first '/' or '|'
        name = raw.split("/")[0].split("|")[0].strip()
    # Remove gene name parenthetical suffix
    name = re.sub(r"\s*\(.*\)\s*$", "", name)
    return name.strip() or raw.strip()[:40]


def parse_homer_denovo(homer_dir: str, n: int = 15) -> pd.DataFrame:
    """
    Parse HOMER de novo results from homerResults.html.

    Parameters
    ----------
    homer_dir : str
        Path to a single HOMER output directory (contains homerResults.html).
    n : int
        Maximum motifs to return, sorted by ascending p-value.

    Returns
    -------
    pd.DataFrame with columns:
        motif_name, pval, qval, neg_log_p, fold_enrichment,
        pct_target, pct_bg, best_known_match, source
    """
    html_path = Path(homer_dir) / "homerResults.html"
    if not html_path.exists():
        return _EMPTY_DF.copy()

    try:
        tables = _extract_tables_from_html(html_path)
    except Exception as exc:
        warnings.warn(f"Could not parse {html_path}: {exc}")
        return _EMPTY_DF.copy()

    # Find the motif results table — typically the largest table,
    # identified by a header row containing "P-value"
    target_table = None
    for tbl in tables:
        if not tbl:
            continue
        header = " ".join(tbl[0]).lower()
        if "p-value" in header or "pvalue" in header:
            target_table = tbl
            break

    if target_table is None or len(target_table) < 2:
        warnings.warn(f"No motif table found in {html_path}")
        return _EMPTY_DF.copy()

    # Identify column indices from header
    header_row = [c.lower().strip() for c in target_table[0]]

    def _find_col(*keywords):
        for i, h in enumerate(header_row):
            if any(kw in h for kw in keywords):
                return i
        return None

    pval_col    = _find_col("p-value", "pvalue")
    logp_col    = _find_col("log p", "log(p")
    pct_t_col   = _find_col("% of target", "target%")
    pct_b_col   = _find_col("% of bg", "% of back", "background%")
    match_col   = _find_col("best match", "match", "known")
    name_col    = _find_col("name")

    if pval_col is None:
        warnings.warn(f"Cannot locate p-value column in {html_path} (header={header_row})")
        return _EMPTY_DF.copy()

    rows = []
    for row in target_table[1:]:
        if len(row) <= max(filter(lambda x: x is not None,
                                  [pval_col, pct_t_col, pct_b_col])):
            continue

        pval = _parse_pval(row[pval_col]) if pval_col < len(row) else np.nan
        if np.isnan(pval):
            continue

        pct_t = _parse_pct(row[pct_t_col]) if pct_t_col is not None and pct_t_col < len(row) else np.nan
        pct_b = _parse_pct(row[pct_b_col]) if pct_b_col is not None and pct_b_col < len(row) else np.nan
        raw_match = row[match_col] if match_col is not None and match_col < len(row) else ""
        raw_name  = row[name_col]  if name_col  is not None and name_col  < len(row) else ""

        motif_name = _clean_motif_name(raw_match) if raw_match else raw_name or f"denovo_{len(rows)+1}"

        fold = (pct_t / max(pct_b, 0.01)) if (not np.isnan(pct_t) and not np.isnan(pct_b)) else np.nan

        rows.append({
            "motif_name":       motif_name,
            "pval":             max(pval, 1e-300),
            "qval":             np.nan,          # HOMER de novo doesn't report q in same column
            "neg_log_p":        -np.log10(max(pval, 1e-300)),
            "fold_enrichment":  fold,
            "pct_target":       pct_t,
            "pct_bg":           pct_b,
            "best_known_match": raw_match.strip()[:80],
            "source":           "denovo",
        })

    if not rows:
        return _EMPTY_DF.copy()

    df = (pd.DataFrame(rows)
            .sort_values("pval")
            .drop_duplicates(subset="motif_name")
            .head(n)
            .reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combined auto-loader
# ─────────────────────────────────────────────────────────────────────────────

def parse_homer_auto(
    homer_dir: str,
    n: int = 15,
    prefer: str = "denovo",
) -> pd.DataFrame:
    """
    Load motif enrichment results from a HOMER directory, trying both
    de novo (homerResults.html) and known (knownResults.txt) sources.

    Parameters
    ----------
    homer_dir : str
        HOMER output directory.
    n : int
        Max motifs to return.
    prefer : {"denovo", "known", "merge"}
        - "denovo": return de novo if available, else fall back to known
        - "known":  return known if available, else fall back to de novo
        - "merge":  concatenate both, deduplicate by motif_name, take top n by p-value

    Returns
    -------
    pd.DataFrame with standard schema plus "source" column ("denovo"/"known").
    """
    from parse_homer_denovo import parse_homer_denovo  # self

    # Lazy import to avoid circular dependency with plot_homer_go_figures
    try:
        from plot_homer_go_figures import parse_homer_known as _parse_known
    except ImportError:
        _parse_known = None

    def _load_known(d, n):
        if _parse_known is None:
            return _EMPTY_DF.copy()
        df = _parse_known(d, n=n)
        if df.empty:
            return _EMPTY_DF.copy()
        df = df.copy()
        if "source" not in df.columns:
            df["source"] = "known"
        missing = [c for c in _EMPTY_DF.columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        return df

    dn_df   = parse_homer_denovo(homer_dir, n=n)
    kn_df   = _load_known(homer_dir, n)

    if prefer == "denovo":
        return dn_df if not dn_df.empty else kn_df
    elif prefer == "known":
        return kn_df if not kn_df.empty else dn_df
    else:  # merge
        combined = pd.concat([dn_df, kn_df], ignore_index=True)
        if combined.empty:
            return _EMPTY_DF.copy()
        return (combined
                .sort_values("pval")
                .drop_duplicates(subset="motif_name")
                .head(n)
                .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Batch loader (all 18 conditions)
# ─────────────────────────────────────────────────────────────────────────────

LAYERS = ["early", "mid", "late"]
PAIRS  = ["blood", "liver", "lymph"]
SIDES  = ["vitro", "vivo"]


def load_all_denovo(
    annotation_dir: str,
    n: int = 200,
    prefer: str = "denovo",
) -> dict:
    """
    Load motif results for all 18 conditions.

    Returns
    -------
    dict: {(layer, side, pair): {motif_name: {neg_log_p, qval, fold_enrichment, source}}}
    """
    homer_base = Path(annotation_dir) / "homer"
    result = {}
    for layer in LAYERS:
        for side in SIDES:
            for pair in PAIRS:
                tag  = f"{layer}_{side}_{pair}"
                hdir = str(homer_base / tag)
                df   = parse_homer_auto(hdir, n=n, prefer=prefer)
                if df.empty:
                    result[(layer, side, pair)] = {}
                else:
                    result[(layer, side, pair)] = {
                        row["motif_name"]: {
                            "neg_log_p":       row["neg_log_p"],
                            "qval":            row["qval"],
                            "fold_enrichment": row["fold_enrichment"],
                            "source":          row.get("source", "unknown"),
                        }
                        for _, row in df.iterrows()
                    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Parse HOMER de novo motif results.")
    ap.add_argument("homer_dir", help="HOMER output directory containing homerResults.html")
    ap.add_argument("--n", type=int, default=15, help="Max motifs to print")
    ap.add_argument("--prefer", choices=["denovo", "known", "merge"], default="denovo")
    args = ap.parse_args()

    df = parse_homer_auto(args.homer_dir, n=args.n, prefer=args.prefer)
    if df.empty:
        print("No results found.", file=sys.stderr)
        sys.exit(1)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", "{:.3g}".format)
    print(df[["motif_name", "pval", "neg_log_p", "fold_enrichment", "source"]].to_string(index=False))
