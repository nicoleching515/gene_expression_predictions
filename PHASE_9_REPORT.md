# PHASE 9 REPORT — HOMER/GO Annotation of Top SAE Features

**Completed:** 2026-04-30

TF motif enrichment and GO:BP enrichment for the top-50 context-divergent SAE features
per layer, using real HOMER (findMotifsGenome.pl, hg38) and g:Profiler REST API.

---

## Method

1. **SAE encoding** — all 12 activation tensors (`activations/{pair}/{cell}/{layer}.pt`,
   shape 10,000 × 1,024) were encoded through the pooled SAE checkpoints
   (`saes/{layer}/pooled.pt`, d\_latent = 8,192).

2. **Top feature selection** — for each of the 18 conditions (3 layers × 3 pairs × 2 sides),
   the top-50 SAE features were selected by absolute per-pair CDS score from
   `results/cds/layer_{layer}_features.tsv`.

3. **BED file generation** — for each top feature, the 300 windows with the highest SAE
   activation were collected and merged into non-overlapping chr8/chr9 intervals. One BED
   file per condition written to `outputs/annotation/homer/{tag}.bed` (289–765 peaks).

4. **HOMER motif enrichment** — `findMotifsGenome.pl {tag}.bed hg38 {outdir} -size 200
   -mask -bg data/windows.bed -p 6` was run for all 18 conditions against the hg38 genome.
   Results written to `outputs/annotation/homer/{tag}/knownResults.txt`.

5. **Gene lookup** — each region was queried against the Ensembl REST API
   (`/overlap/region/human/{region}?feature=gene;biotype=protein_coding`)
   to retrieve overlapping protein-coding genes.

6. **GO:BP enrichment** — gene lists were submitted to the g:Profiler REST API
   (source `GO:BP`, FDR < 0.05, BH correction). Results written to
   `outputs/annotation/go/{tag}_go.tsv`.

7. **Figure generation** — `src/plot_homer_go_figures.py` was run to produce
   fig8 (TF motif heatmap) and fig9 (GO:BP dot-plot).

---

## Layer: EARLY

### Top HOMER Motifs (top 2 by p-value)

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | PBX1, Foxa2 |
| blood | vitro | RORgt, RORgt |
| liver | vivo | NF1-halfsite, SpiB |
| liver | vitro | MafA, Phox2b |
| lymph | vivo | E2F, RXR |
| lymph | vitro | Sox3, RAR:RXR |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | transcription by RNA polymerase II; regulation of transcription by RNA polymerase II |
| blood | vitro | positive regulation of transcription by RNA polymerase II; positive regulation of DNA-templated transcription |
| liver | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| liver | vitro | regulation of DNA-templated transcription; regulation of RNA biosynthetic process |
| lymph | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| lymph | vitro | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |

---

## Layer: MID

### Top HOMER Motifs (top 2 by p-value)

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | Gsx2, MITF |
| blood | vitro | HOXA3, CTCF-SatelliteElement |
| liver | vivo | Gli2, VDR |
| liver | vitro | Pitx1:Ebox, Pit1 |
| lymph | vivo | Oct6, Brn1 |
| lymph | vitro | ZNF382, KLF6 |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | transcription by RNA polymerase II; regulation of transcription by RNA polymerase II |
| blood | vitro | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| liver | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| liver | vitro | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| lymph | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| lymph | vitro | regulation of RNA biosynthetic process; regulation of DNA-templated transcription |

---

## Layer: LATE

### Top HOMER Motifs (top 2 by p-value)

| Pair | Side | Top motifs |
|------|------|------------|
| blood | vivo | RUNX2, RUNX-AML |
| blood | vitro | Tcfcp2l1, TFE3 |
| liver | vivo | NFkB-p65-Rel, Pax7 |
| liver | vitro | FOXA1, Nkx2.5 |
| lymph | vivo | ZBED2, NFkB-p65-Rel |
| lymph | vitro | Tlx?, ERb |

### Top GO:BP Terms

| Pair | Side | Top GO terms |
|------|------|--------------|
| blood | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| blood | vitro | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| liver | vivo | transcription by RNA polymerase II; regulation of transcription by RNA polymerase II |
| liver | vitro | transcription by RNA polymerase II; regulation of transcription by RNA polymerase II |
| lymph | vivo | regulation of transcription by RNA polymerase II; transcription by RNA polymerase II |
| lymph | vitro | positive regulation of transcription by RNA polymerase II; negative regulation of transcription by RNA polymerase II |

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
| Fig 9 | `results/figures/fig9_go_enrichment.{pdf,png}` | GO:BP dot-plot (dot size = fold enrichment, colour = −log₁₀ FDR) |

---

## Scripts

| Script | Description |
|--------|-------------|
| `generate_annotation_data.py` | SAE encoding → Ensembl gene lookup → g:Profiler GO:BP enrichment → writes annotation outputs |
| `run_annotation_v2.sh` | Full HOMER pipeline: BED generation → findMotifsGenome.pl → GO enrichment |
| `src/plot_homer_go_figures.py` | Generates fig8 and fig9 from HOMER knownResults.txt and GO TSV files |
