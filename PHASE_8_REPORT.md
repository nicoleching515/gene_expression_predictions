# PHASE 8 REPORT — Bio Annotation of Top SAE Features

HOMER motif enrichment (findMotifsGenome.pl, hg38, -size 200 -mask) and GO:BP enrichment
(g:Profiler REST API, FDR < 0.05) for the top-50 context-divergent SAE features per layer.
Background: all 10,000 genomic windows (`data/windows.bed`).

---

## Method

1. **BED file generation** — for each of the 18 conditions (3 layers × 3 pairs × 2 sides),
   the top-50 SAE features by absolute CDS score were selected. The 300 windows with highest
   SAE activation per feature were merged into non-overlapping intervals, producing one BED
   file per condition (289–765 peaks).

2. **HOMER motif enrichment** — `findMotifsGenome.pl {condition}.bed hg38 {outdir} -size 200
   -mask -bg data/windows.bed -p 6` was run for all 18 conditions. Results written to
   `outputs/annotation/homer/{condition}/knownResults.txt`.

3. **GO:BP enrichment** — overlapping protein-coding genes were retrieved via Ensembl REST API
   and submitted to g:Profiler (GO:BP, FDR < 0.05, BH correction). Results written to
   `outputs/annotation/go/{condition}_go.tsv`.

4. **Figure generation** — `src/plot_homer_go_figures.py` was run to produce fig8 (TF motif
   heatmap) and fig9 (GO:BP dot-plot).

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

## Peak counts per condition

| Layer | Pair | Side | Peaks |
|-------|------|------|-------|
| early | blood | vitro | 488 |
| early | blood | vivo | 330 |
| early | liver | vitro | 518 |
| early | liver | vivo | 289 |
| early | lymph | vitro | 396 |
| early | lymph | vivo | 470 |
| mid | blood | vitro | 458 |
| mid | blood | vivo | 449 |
| mid | liver | vitro | 513 |
| mid | liver | vivo | 606 |
| mid | lymph | vitro | 670 |
| mid | lymph | vivo | 415 |
| late | blood | vitro | 765 |
| late | blood | vivo | 611 |
| late | liver | vitro | 690 |
| late | liver | vivo | 587 |
| late | lymph | vitro | 713 |
| late | lymph | vivo | 541 |

---

## Figures

| Figure | Path | Description |
|--------|------|-------------|
| Fig 8 | `results/figures/fig8_homer_motifs.{pdf,png}` | TF motif enrichment heatmap (−log₁₀ p) across all 18 conditions |
| Fig 9 | `results/figures/fig9_go_enrichment.{pdf,png}` | GO:BP dot-plot (dot size = fold enrichment, colour = −log₁₀ FDR) |
