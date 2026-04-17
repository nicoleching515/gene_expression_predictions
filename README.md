# EpiBERT Contrastive SAE Analysis

## About

This project implements a mechanistic interpretability pipeline for **EpiBERT**, a transformer-based chromatin accessibility model. The central question is: **do EpiBERT's internal representations encode distinct features for in vitro (cell-line) vs. in vivo (tissue) chromatin contexts?**

We use **Sparse Autoencoders (SAEs)** with a BatchTopK activation function to decompose EpiBERT's hidden states into interpretable latent features, then apply contrastive analysis across three matched cell-line / tissue pairs:

| Pair | In vitro (cell line) | In vivo (tissue) |
|------|----------------------|------------------|
| blood | K562 | Hematopoietic Stem Cell |
| liver | HepG2 | Liver Tissue |
| lymph | GM12878 | Naive B Cell |

The pipeline collects activations at three depths (early L/4, mid L/2, late 3L/4), trains one SAE per layer per context regime (9 total), computes a **Context Divergence Score (CDS)** per feature, then validates discovered features with causal ablation and context-steering experiments.

This work is being prepared for ICML submission.

---

## Environment Recreation

### 1. Clone the repository

```bash
git clone https://github.com/nicoleching515/gene_expression_predictions.git /workspace
cd /workspace
```

> Git LFS is required for large pointer files. Install it first:
> ```bash
> git lfs install
> git lfs pull
> ```

### 2. Set up Python environment

```bash
cd /workspace/project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/saprmarks/dictionary_learning.git
```

### 3. Verify GPU

```python
import torch
assert torch.cuda.is_available(), "No GPU detected"
print(torch.cuda.get_device_name(0))  # expected: H100
```

### 4. Configure Weights & Biases (for SAE training)

```bash
wandb login
# or run offline:
export WANDB_MODE=offline
```

---

## Required Data Files

All large files live directly under `/workspace/`. Re-download them from ENCODE (accession links below) and place them at the exact paths shown.

### ATAC-seq BAM files

| File | Cell / Tissue | Size |
|------|--------------|------|
| `K562-ATAC.bam` + `.bai` | K562 (blood cell line) | ~10 GB |
| `HepG2-ATAC.bam` + `.bai` | HepG2 (liver cell line) | ~9 GB |
| `GM12878-ATAC.bam` + `.bai` | GM12878 (lymphoblast cell line) | ~9.2 GB |
| `Hemapoetic Stem Cell-ATAC.bam` + `.bai` | HSC (in vivo blood) | ~1.1 GB |
| `Liver Tissue-ATAC.bam` + `.bai` | Liver Tissue (in vivo) | ~17 GB |
| `Naive B Cell-ATAC.bam` + `.bai` | Naive B Cell (in vivo lymph) | ~2.2 GB |

Search [ENCODE](https://www.encodeproject.org/) by biosample name and assay type `ATAC-seq`. Download aligned BAM files (hg38 assembly).

### WGS BAM/CRAM files (cell lines only)

| File | Cell line | Size |
|------|-----------|------|
| `K562-WGS.bam` + `.bai` | K562 | ~337 GB |
| `HepG2-WGS.bam` + `.bai` | HepG2 | ~187 GB |
| `GM12878-WGS.cram` + `.crai` | GM12878 | ~15.8 GB |

Also from ENCODE; assay type `WGS`. These are only required if re-running Phase 1 (data pipeline). The processed `.npy` arrays in `project/data/atac_processed/` are sufficient for Phases 2–9.

### Reference genome

```bash
mkdir -p /workspace/project/data/genome
cd /workspace/project/data/genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa
```

### EpiBERT model checkpoints

The pre-trained and fine-tuned EpiBERT weights live in `/workspace/epibert_models/`. If they are not restored from the repo (they may be LFS-tracked), obtain them from the EpiBERT authors and place them at:

```
epibert_models/pretrained/model1/ckpt-45.*
epibert_models/pretrained/model2/ckpt-45.*
epibert_models/fine_tuned/ckpt-16.*
```

---

## Running the Pipeline

All hyperparameters are controlled by `project/configs/main.yaml`. Do not hard-code values in scripts.

### Reproduce figures from cached results (Phases 4–7)

```bash
cd /workspace/project
source .venv/bin/activate
bash reproduce.sh
```

### Run the full pipeline from scratch

```bash
# Phase 1: Data pipeline (CPU ok)
python project/src/data.py

# Phase 2: Collect activations (GPU required)
python project/src/train_sae.py --phase collect

# Phase 3: Train SAEs (GPU required, ~50K steps × 9 SAEs)
python project/src/train_sae.py --phase train

# Phases 4–7: Analysis, ablation, steering, figures
python project/src/run_pipeline.py --start-phase 4
```

Logs are written to `project/logs/`. Results land in `project/results/` and top feature exports go to `outputs/`.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pipeline & ATAC normalization | Complete |
| 2 | Activation collection (60K vectors, 3 layers) | Complete |
| 3 | SAE training (9 SAEs) | Complete |
| 4 | Contrastive analysis & CDS | Complete |
| 5 | Causal ablation experiments | Complete |
| 6 | Context steering experiments | Complete |
| 7 | Figures & statistical tests | Complete |
| 8 | Bio annotation figures (HOMER, GO, ChromHMM) | Pending bio team |
| 9 | Reproducibility pass & git tag | Pending |

---

## Key Results (Phases 1–7)

- **CDS signal:** 0.70% → 1.00% → 2.62% of features pass Bonferroni correction at early / mid / late layers — context specificity increases with depth.
- **Ablation:** Targeted ablation of vivo-enriched features produces Cohen's d = 1.789 vs. random ablation (Wilcoxon p = 2.98×10⁻⁸).
- **Steering gap closure:** Median = 0.041 (4.5× above random baseline); best sweep (α=2.0, β=0.5) reaches 0.112.
- **Cross-layer overlap:** Jaccard ≤ 0.02 across all layer pairs — features are hierarchically non-redundant.
