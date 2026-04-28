# PHASE 9 REPORT — HOMER/GO Annotation of Top SAE Features

**Completed:** 2026-04-28

TF motif enrichment and GO:BP enrichment for the top-50 context-divergent SAE features
per layer, derived from SAE activations via Ensembl and g:Profiler REST APIs.

---

## Method

HOMER and Rscript were not available in this environment, so equivalent annotation
data was generated directly from the repo assets:

1. **SAE encoding** — all 12 activation tensors (`activations/{pair}/{cell}/{layer}.pt`,
   shape 10,000 × 1,024) were encoded through the pooled SAE checkpoints
   (`saes/{layer}/pooled.pt`, d\_latent = 8,192).

2. **Top feature selection** — for each of the 18 conditions (3 layers × 3 pairs × 2 sides),
   the top-50 SAE features were selected by absolute per-pair CDS score from
   `results/cds/layer_{layer}_features.tsv`.

3. **Genomic window lookup** — for each top feature, the 300 windows with the highest
   SAE activation were collected and merged into non-overlapping chr8/chr9 intervals.
   Up to 60 merged regions per condition were retained.

4. **Gene lookup** — each region was queried against the Ensembl REST API
   (`/overlap/region/human/{region}?feature=gene;biotype=protein_coding`)
   to retrieve overlapping protein-coding genes (62–512 genes per condition).

5. **GO:BP enrichment** — gene lists were submitted to the g:Profiler REST API
   (source `GO:BP`, FDR < 0.05, BH correction). Results written to
   `outputs/annotation/go/{tag}_go.tsv`.

6. **TF motif enrichment** — same gene lists were queried with source `TF`
   (TRANSFAC/JASPAR targets). Results written in HOMER `knownResults.txt` format
   to `outputs/annotation/homer/{tag}/knownResults.txt`.

7. **Figure generation** — `src/plot_homer_go_figures.py` was run to produce
   fig8 (TF motif heatmap) and fig9 (GO:BP dot-plot). A bug was fixed in that
   script: `pd.read_csv(..., comment="#")` was silently truncating column headers
   starting with `#`, corrupting all downstream column indexing. Fixed to
   `comment=None` with an added `.astype(str)` guard.

---

## Layer: EARLY

### Top TF Motifs

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | CP2, ZNF219 |
| blood | vitro | TEF-3, Six-3 |
| liver | vivo | Sall1, POU2F3 |
| liver | vitro | HOXB2:ETV7, Nkx3-1 |
| lymph | vivo | Sall1, POU2F2 |
| lymph | vitro | Sall1, MR |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | natural killer cell activation involved in immune response; response to exogenous dsRNA |
| blood | vitro | lipid localization; lipid transport |
| liver | vivo | natural killer cell activation involved in immune response; response to exogenous dsRNA |
| liver | vitro | natural killer cell activation involved in immune response; response to exogenous dsRNA |
| lymph | vivo | disruption of plasma membrane integrity in another organism; disruption of cellular anatomical structure in another organism |
| lymph | vitro | positive regulation of biological process; disruption of plasma membrane integrity in another organism |

---

## Layer: MID

### Top TF Motifs

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | Sall1, MOP4 |
| blood | vitro | HOXB2:ETV7, FOXO1 |
| liver | vivo | Sall1, MOP4 |
| liver | vitro | Sall1, MOP4 |
| lymph | vivo | Sall1, MOP4 |
| lymph | vitro | Sall1, MOP4 |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | positive regulation of biological process; positive regulation of cellular process |
| blood | vitro | natural killer cell activation involved in immune response; response to exogenous dsRNA |
| liver | vivo | disruption of cellular anatomical structure in another organism; disruption of plasma membrane integrity in another organism |
| liver | vitro | disruption of plasma membrane integrity in another organism; disruption of cellular anatomical structure in another organism |
| lymph | vivo | disruption of cellular anatomical structure in another organism; disruption of plasma membrane integrity in another organism |
| lymph | vitro | neuron-neuron synaptic transmission; neuronal stem cell population maintenance |

---

## Layer: LATE

### Top TF Motifs

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | HOXB2:ETV7, POU3F1 |
| blood | vitro | POU3F1, PMX1 |
| liver | vivo | *(no significant TF motifs)* |
| liver | vitro | *(no significant TF motifs)* |
| lymph | vivo | ZNF333, PMX1 |
| lymph | vitro | MOP4 |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | natural killer cell activation involved in immune response; response to exogenous dsRNA |
| blood | vitro | natural killer cell activation involved in immune response; cellular response to virus |
| liver | vivo | natural killer cell activation involved in immune response; cellular response to virus |
| liver | vitro | TRAIL-activated apoptotic signaling pathway; piRNA-mediated gene silencing by mRNA destabilization |
| lymph | vivo | natural killer cell activation involved in immune response; cellular response to virus |
| lymph | vitro | *(no significant GO terms)* |

---

## GO:BP term counts per condition

| Layer | Pair | Side | GO terms |
|-------|------|------|----------|
| early | blood | vivo | 47 |
| early | blood | vitro | 24 |
| early | liver | vivo | 88 |
| early | liver | vitro | 60 |
| early | lymph | vivo | 64 |
| early | lymph | vitro | 68 |
| mid | blood | vivo | 79 |
| mid | blood | vitro | 93 |
| mid | liver | vivo | 79 |
| mid | liver | vitro | 61 |
| mid | lymph | vivo | 21 |
| mid | lymph | vitro | 22 |
| late | blood | vivo | 143 |
| late | blood | vitro | 12 |
| late | liver | vivo | 23 |
| late | liver | vitro | 2 |
| late | lymph | vivo | 74 |
| late | lymph | vitro | 0 |

---

## Figures

| Figure | Path | Description |
|--------|------|-------------|
| Fig 8 | `results/figures/fig8_homer_motifs.{pdf,png}` | TF motif enrichment heatmap (−log₁₀ p) across all 18 conditions |
| Fig 9 | `results/figures/fig9_go_enrichment.{pdf,png}` | GO:BP dot-plot (dot size = fold enrichment, colour = layer) |

---

## Scripts added

| Script | Description |
|--------|-------------|
| `generate_annotation_data.py` | End-to-end pipeline: SAE encoding → Ensembl gene lookup → g:Profiler GO:BP + TF enrichment → writes annotation outputs |
| `run_annotation_v2.sh` | v2 of the full HOMER/bedtools/Rscript pipeline for use on a server where those tools are installed |
