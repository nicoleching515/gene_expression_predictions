# PHASE 11 REPORT — ENCODE ATAC-seq Genome-Wide HOMER and Fig 11

**Completed:** 2026-04-30

Extended the HOMER motif enrichment analysis from SAE-derived chr8/chr9 windows to
genome-wide ENCODE ATAC-seq IDR peak sets for all 6 cell-type conditions, including
a mm10→hg38 liftover for HSC (no human ENCODE data available). Generated Fig 11.

---

## Motivation

Phase 10 ran HOMER genome-wide using SAE feature–derived BED files (chr8/chr9 only,
289–765 peaks per condition). While the background was fixed (−genomeBg), the target
regions were still chromosome-restricted. Phase 11 replaces targets with full-genome
ENCODE ATAC-seq IDR peaks (30k–110k peaks per condition), providing substantially
more statistical power (p-values now reach 1e-1367 vs. 1e-3 before).

---

## Cell Types and ENCODE Files

| Condition          | Label                   | ENCODE Accession   | Peaks   |
|--------------------|-------------------------|--------------------|---------|
| K562_specific_blood  | K562 (erythroid, vs HSC)  | ENCFF738NOA / ENCSR956DNB | 30,829  |
| HSC_specific_blood   | HSC (stem cell, vs K562)  | ENCFF395WJI (mm10→hg38 liftover) | 35,647  |
| HepG2_vitro_liver    | HepG2 (hepatoma, vs Liver)| ENCFF791RKW / ENCSR042AWH | 101,542 |
| Liver_vivo_liver     | Liver tissue (vs HepG2)   | ENCFF654SUU / ENCSR124NNL | 101,793 |
| GM12878_vitro_lymph  | GM12878 (EBV-B, vs NaiveB)| ENCFF917REN / ENCSR095QNB | 110,610 |
| NaiveB_vivo_lymph    | Naive B cell (vs GM12878) | ENCFF380TOL + ENCFF590QLY (2 donors merged) | 44,826 |

**Note on HSC:** No human HSC ATAC-seq IDR peaks exist on ENCODE. Mouse ENCSR366VBB
(ENCFF395WJI, mm10) was lifted over to hg38 using pyliftover (46% liftover rate,
106,082 mm10 → 30,459 hg38 peaks after merge). Condition-specific peaks were computed
against K562 (bedtools subtract), yielding 35,647 HSC-specific peaks.

---

## HOMER Method

```bash
findMotifsGenome.pl {condition_specific.bed} hg38 {outdir} \
    -size 200 -mask -genomeBg -p 8
```

Condition-specific peaks computed with `bedtools subtract` for each cell-type pair.
HOMER version: 5.1; genome: hg38; background: ~100,000 GC-matched random sequences.

---

## Results: Top Motifs per Condition

### K562 (erythroid, vitro blood — K562-specific vs HSC)
| Motif | p-value | % Target | % Background |
|-------|---------|----------|--------------|
| Gata6 | 1e-544 | 3.76% | — |
| Gata2 | 1e-530 | 2.81% | — |
| Gata1 | 1e-528 | 2.55% | — |
| Gata4 | 1e-500 | 4.34% | — |
| GATA3 | 1e-458 | 6.52% | — |
| Jun-AP1 | 1e-408 | 1.08% | — |
| Fosl2 | 1e-394 | 1.53% | — |
| TRPS1 | 1e-389 | 8.63% | — |

GATA1/2 are master regulators of erythropoiesis — precisely the expected hallmark of
K562 (chronic myeloid leukemia, erythroid-megakaryocyte lineage).

### HSC (vivo blood — HSC-specific, mm10→hg38 liftover)
| Motif | p-value | % Target | % Background |
|-------|---------|----------|--------------|
| CTCF | 1e-1367 | 0.93% | — |
| BORIS | 1e-764 | 2.02% | — |
| EWS:ERG-fusion | 1e-134 | 5.99% | — |
| Etv2 | 1e-130 | 8.43% | — |
| ERG | 1e-111 | 15.30% | — |
| PU.1 | 1e-109 | 4.16% | — |
| ETS1 | 1e-101 | 9.84% | — |
| ELF3 | 1e-98 | 6.31% | — |

CTCF/BORIS (CTCFL) dominate, consistent with strong insulator/TAD boundary activity in
HSC-specific open chromatin (liftover peaks likely capture conserved CTCF-bound elements).
ERG, ETV2, and PU.1 are ETS-family TFs with established roles in haematopoietic stem
cell specification and myeloid/lymphoid lineage priming.

### HepG2 (vitro liver — hepatoma cell line, vs Liver tissue)
| Motif | p-value |
|-------|---------|
| FOXA1 | 1e-229 |
| HNF4a | 1e-228 |
| FOXM1 | 1e-206 |
| Fox:Ebox | 1e-185 |
| CTCF | 1e-166 |
| Foxa2 / Foxa3 | 1e-157/1e-149 |

FOXA1 ("FoxA pioneer factor") and HNF4α are the canonical master regulators of hepatocyte
identity. Their strong enrichment in HepG2-specific peaks confirms that HepG2, despite
being a hepatoma, retains hepatocyte-lineage TF binding patterns.

### Liver tissue (vivo liver — primary tissue, vs HepG2)
| Motif | p-value |
|-------|---------|
| CTCF | 1e-1267 |
| Fra1 | 1e-910 |
| Fos | 1e-884 |
| Fra2 | 1e-882 |
| JunB | 1e-867 |
| Atf3 | 1e-851 |
| Fosl2 | 1e-828 |
| BATF | 1e-823 |

AP-1 family (Fos/Jun/ATF) dominates primary liver-specific peaks. This is consistent
with AP-1 being a general "open chromatin" marker in primary tissue (stress-response,
inflammatory signalling); the contrast with HepG2 makes biological sense because cell
lines have diminished AP-1 activity relative to primary tissues.

### GM12878 (vitro lymph — EBV-transformed B cell, vs Naive B)
| Motif | p-value |
|-------|---------|
| BATF | 1e-632 |
| JunB | 1e-615 |
| Fra1 | 1e-589 |
| Atf3 | 1e-578 |
| Fos | 1e-573 |
| CTCF | 1e-544 |
| AP-1 | 1e-505 |

AP-1/BATF signature is a hallmark of EBV-driven B-cell transformation — EBV LMP1
activates AP-1 and NF-κB, explaining the strong BATF/JunB/Fra1 enrichment in GM12878
relative to naive B cells.

### Naive B cell (vivo lymph — primary, vs GM12878)
| Motif | p-value |
|-------|---------|
| CTCF | 1e-406 |
| PU.1 | 1e-382 |
| SpiB | 1e-372 |
| ELF5 | 1e-298 |
| ELF3 | 1e-244 |
| Elf4 | 1e-220 |
| BORIS | 1e-198 |
| ETS1 | 1e-177 |

PU.1/SpiB/ELF-family ETS factors are the canonical regulators of B-cell development
and naive B-cell identity. This is the expected ground-truth result — naive B cells
have prominent ETS-factor binding at accessible chromatin relative to the AP-1-dominated
GM12878 (EBV-activated) profile.

---

## Key Biological Findings

1. **GATA1/2 in K562 confirms erythroid specificity** — GATA factors are the textbook
   master regulators of red blood cell lineage. Their overwhelming enrichment (p=1e-528
   to 1e-544) validates the ENCODE K562 approach.

2. **ETS factors in HSC (PU.1, ERG, ETV2)** — Consistent with haematopoietic stem cell
   chromatin landscape. PU.1 is a pioneer factor for myeloid/lymphoid lineage priming;
   ERG is essential for HSC maintenance.

3. **FOXA1/HNF4α in HepG2 vs. AP-1 in Liver tissue** — A clear vitro/vivo contrast:
   cell lines preserve lineage-TF binding (FOXA1, hepatocyte identity) while primary tissue
   is enriched for AP-1 (stress/inflammatory chromatin states).

4. **AP-1/BATF in GM12878 vs ETS in Naive B** — EBV transformation (GM12878) hijacks
   AP-1/BATF signalling, while primary naive B cells show PU.1/SpiB occupancy as expected
   for committed B-lineage cells.

5. **CTCF ubiquity in "vivo" and "specific" sets** — CTCF/BORIS dominate several
   condition-specific peak sets (HSC, Liver, NaiveB), suggesting these sets capture
   topological domain boundaries that differ between paired cell types.

---

## Figure

**Fig 11** (`results/figures/fig11_encode_atac_homer.{pdf,png}`): Heatmap of
−log₁₀(p-value) for top-20 TF motifs across all 6 ENCODE ATAC-seq conditions.
Bubble size = fold enrichment (% target / % background). ★ = FDR q < 0.05.
Y-axis coloured red (vitro/cell line) vs. blue (vivo/primary tissue).

---

## Files Added

| File | Description |
|------|-------------|
| `run_homer_encode_atac.sh` | Downloads ENCODE ATAC peaks, lifts over HSC mm10→hg38, computes specific peaks, runs HOMER |
| `src/plot_encode_atac_figures.py` | Generates fig11 heatmap from ENCODE HOMER results |
| `outputs/annotation/homer_encode/peaks/` | Condition-specific BED files (6 conditions) |
| `outputs/annotation/homer_encode/homer/` | HOMER output directories (6 conditions) |
| `results/figures/fig11_encode_atac_homer.pdf` | Publication-quality vector figure |
| `results/figures/fig11_encode_atac_homer.png` | 300 dpi raster figure |
| `PHASE_11_REPORT.md` | This report |
