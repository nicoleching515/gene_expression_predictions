# PHASE 12 REPORT — HOMER De Novo Integration & Multi-Model Generalization

**Status:** Implementation complete — pending GPU runs  
**Date:** 2026-05-02

---

## Motivation

Phase 11 demonstrated that ENCODE ATAC-seq–based HOMER analysis yields highly
significant TF motifs (p-values reaching 1e−1367) when run on real genome-wide
peaks.  Two gaps remained for a NeurIPS-tier submission:

1. **HOMER de novo results were unused.**  All previous figures (Figs 8–11)
   relied exclusively on `knownResults.txt` — enrichment of a pre-compiled
   TF motif database.  HOMER's *de novo* motif discovery (`homerResults.html`)
   identifies motifs directly from the data without database bias and is the
   more rigorous analysis for novel biological claims.

2. **Single-model scope.**  All interpretability results (CDS, ablation,
   steering) came from EpiBERT alone.  A NeurIPS paper requires demonstrating
   that the SAE contrastive analysis is **model-agnostic** — i.e. the same
   vitro/vivo chromatin dichotomy is encoded (or not) in competing sequence
   models.

---

## Changes

### A. HOMER De Novo Integration

#### New files
| File | Description |
|------|-------------|
| `src/parse_homer_denovo.py` | Parser for HOMER `homerResults.html` (de novo discovered motifs). Extracts motif name (best known TF match), p-value, % target, % background, fold enrichment. API-compatible with `parse_homer_known()`. |
| `run_homer_with_denovo.sh` | Replaces `run_homer_genomewide.sh`. Drops `-nomotif` flag so HOMER runs both de novo and known-motif enrichment. Preserves `homerResults.html` and archives per-motif `.motif` files into `denovo_motifs.tar.gz`. Writes a run manifest at `outputs/annotation/homer/homer_manifest.tsv`. |

#### Modified files
| File | Change |
|------|--------|
| `src/plot_homer_go_figures.py` | Added `--homer-source` argument (`known` / `denovo` / `auto_denovo` / `auto_known` / `merge`). Default changed to `auto_denovo`: prefer de novo HTML results, fall back to `knownResults.txt`. New helper `_load_homer_condition()` dispatches on source. Figure title and footer now reflect the active source. |
| `configs/main.yaml` | Added `homer:` block with `source`, `n_denovo`, and `run_script` settings. |

#### How to reproduce

```bash
# 1. Re-run HOMER with de novo enabled (keeps homerResults.html)
cd /workspace/gene_expression_predictions
bash run_homer_with_denovo.sh --threads 8 --n-denovo 25

# 2. Regenerate Fig 8 using de novo motifs
python src/plot_homer_go_figures.py \
    --annotation_dir outputs/annotation \
    --figures_dir    results/figures \
    --homer-source   denovo

# 3. Or merge known + de novo
python src/plot_homer_go_figures.py \
    --annotation_dir outputs/annotation \
    --figures_dir    results/figures \
    --homer-source   merge
```

---

### B. Multi-Model Generalisation

Four DNA sequence models are now supported in the contrastive SAE pipeline:

| Model | Architecture | Hidden dim | Layers | Hook depths | DNA input |
|-------|-------------|-----------|--------|-------------|----------|
| **EpiBERT** | Performer transformer + ATAC | 1,024 | 8 | L/4=2, L/2=4, 3L/4=6 | hg38 + ATAC BAM |
| **Enformer** | Conv stem + 11 transformer blocks | 1,536 | 11 | 2, 5, 8 | hg38 (196,608 bp) |
| **HyenaDNA** | Hyena long-conv operator | 256 | 8 | 1, 3, 5 | hg38 (131,072 bp) |
| **Nucleotide Transformer v1** | ESM-style BERT (6-mer tokens) | 1,280 | 24 | 6, 12, 18 | hg38 (3,072 bp) |

All models use actual hg38 reference sequence (`--use-genome`) over 9,004 blacklist-filtered windows
distributed across chr1–chr22 + chrX, matching the scope of the HOMER genome-wide analysis.

#### New files

| File | Description |
|------|-------------|
| `src/models/__init__.py` | Adapter registry (`register`, `get_adapter`, `list_adapters`). |
| `src/models/base.py` | Abstract `ModelAdapter` base class with `get_activations()` interface. |
| `src/models/epibert_adapter.py` | EpiBERT adapter (wraps existing `model_torch.py`). |
| `src/models/enformer_adapter.py` | Enformer adapter (`enformer-pytorch`; hooks transformer blocks 2/5/8). |
| `src/models/hyenadna_adapter.py` | HyenaDNA adapter (HuggingFace `LongSafari`; hooks Hyena layers). |
| `src/models/nucleotide_transformer_adapter.py` | NT adapter (HuggingFace `InstaDeepAI`; hooks BERT encoder layers). |
| `src/collect_activations_multimodel.py` | Collects activations for all four models over all 6 conditions × 10K windows. Outputs to `activations/{model}/{pair}/{condition}/{layer}.pt`. |
| `src/cross_model_analysis.py` | Trains one SAE per (model, layer), computes CDS per pair, generates cross-model summary and Jaccard tables in `results/cross_model/`. |
| `src/plot_cross_model_figures.py` | Generates Figs 12–14 (see below). |

#### Modified files
| File | Change |
|------|--------|
| `configs/main.yaml` | Added `multi_model:` block with per-model pretrained IDs, hidden dims, hook layers, batch sizes, and SAE hyperparameters. |

#### New figures

| Figure | File | Description |
|--------|------|-------------|
| **Fig 12** | `fig12_cross_model_cds.{pdf,png}` | Grouped bar chart: % Bonferroni-significant context-divergent features per model × layer × cell-type pair. Directly tests whether each model's internal representations encode the vitro/vivo chromatin distinction. |
| **Fig 13** | `fig13_cross_model_jaccard.{pdf,png}` | Heatmap of Jaccard similarity between top-50 context-divergent feature sets across model pairs (per layer, per cell-type pair). Low Jaccard → models discover complementary features; high Jaccard → shared representational structure. |
| **Fig 14** | `fig14_cross_model_cds_violin.{pdf,png}` | Violin plots of \|CDS\| magnitude distribution per model × layer. Shows whether context-specificity grows with depth uniformly or differently across architectures. |

#### Installation of new dependencies

```bash
pip install enformer-pytorch          # Enformer (lucidrains re-implementation)
pip install transformers accelerate   # HyenaDNA + Nucleotide Transformer
pip install sentencepiece             # NT tokeniser
```

#### How to run

```bash
# Step 1: Collect activations for all 4 models
python src/collect_activations_multimodel.py --skip-existing

# Step 2: Train SAEs + compute CDS for all models
python src/cross_model_analysis.py

# Step 3: Generate comparison figures
python src/plot_cross_model_figures.py \
    --results_dir results/cross_model \
    --figures_dir results/figures

# Quick smoke-test (100 windows, 200-step SAEs):
python src/collect_activations_multimodel.py --n-windows 100 --model hyenadna
python src/cross_model_analysis.py --smoke-test
```

---

## Expected Research Outcomes (NeurIPS claim)

The multi-model pipeline enables the following new claims:

1. **Model-agnostic vitro/vivo feature detection** — if multiple sequence models
   all develop Bonferroni-significant context-divergent SAE features, it suggests
   the vitro/vivo distinction is a genuine signal in DNA sequence context rather
   than an artifact of EpiBERT's specific training objective.

2. **Architecture comparison** — EpiBERT (with explicit ATAC input) should
   show the strongest context-specificity; Enformer (multi-track output
   implicitly supervised on ATAC) should be second; sequence-only models
   (HyenaDNA, NT) may show weaker but non-zero context signals from
   sequence composition alone.

3. **De novo vs. known motif contrast** — HOMER de novo motifs that are *not*
   in the known TF database but are enriched specifically in context-divergent
   SAE feature windows represent novel biological discoveries about vitro/vivo
   chromatin regulatory differences.

---

## Files Added / Modified

| File | Status |
|------|--------|
| `src/parse_homer_denovo.py` | New |
| `run_homer_with_denovo.sh` | New |
| `src/models/__init__.py` | New |
| `src/models/base.py` | New |
| `src/models/epibert_adapter.py` | New |
| `src/models/enformer_adapter.py` | New |
| `src/models/hyenadna_adapter.py` | New |
| `src/models/nucleotide_transformer_adapter.py` | New |
| `src/collect_activations_multimodel.py` | New |
| `src/cross_model_analysis.py` | New |
| `src/plot_cross_model_figures.py` | New |
| `src/plot_homer_go_figures.py` | Modified (--homer-source flag, de novo dispatch) |
| `configs/main.yaml` | Modified (multi_model + homer blocks) |
| `PHASE_12_REPORT.md` | New |
