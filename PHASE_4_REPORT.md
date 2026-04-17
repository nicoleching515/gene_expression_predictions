# PHASE 4 REPORT — Contrastive Analysis

**Completed:** 2026-04-17  
**Wall clock:** ~1 min  
**Layers processed:** early (L/4), mid (L/2), late (3L/4)

---

## 1. CDS Distribution Summary

| Layer | Features | CDS avg mean | CDS avg std | Max |CDS| | Bonferroni sig | Sign-consistent |
|---|---|---|---|---|---|---|
| early | 8,192 | -0.0014 | 0.195 | 8.87 | 0.70% (57) | 0.1% |
| mid   | 8,192 | -0.0015 | 0.229 | 8.30 | 1.00% (82) | 0.2% |
| late  | 8,192 | -0.0447 | 0.657 | 24.11 | 2.62% (215) | 1.1% |

**Key observation — layer gradient:** The CDS signal grows substantially with depth. Standard deviation of the CDS distribution increases 3.4× from early to late, and the number of Bonferroni-significant features nearly quadruples (57 → 215). This is consistent with the Phase 2 activation L2 divergence results (35–108 in early → 165–200 in late): the model builds increasingly distinct representations as context propagates through layers.

**Vitro bias:** At every layer, `vitro_enriched` features outnumber `vivo_enriched` features by ~1.5× (early: 55 vs 36; mid: 114 vs 71; late: 820 vs 602). This likely reflects that cell-line ATAC profiles (K562, HepG2, GM12878) are more extreme/uniform than primary tissue — higher signal-to-noise in BAM coverage, which may drive stronger activation in certain SAE features.

**Sign consistency is low (0.1–1.1%):** Very few features diverge in the same direction across all 3 pairs (blood, liver, lymph). This means most context-divergent features are pair-specific rather than universal vivo/vitro markers. Cross-pair agreement will be an important framing point for reviewers.

---

## 2. Feature Categories by Layer

| Category | early | mid | late |
|---|---|---|---|
| other | 8,101 (98.9%) | 8,007 (97.7%) | 6,769 (82.6%) |
| vitro_enriched | 55 (0.7%) | 114 (1.4%) | 820 (10.0%) |
| vivo_enriched | 36 (0.4%) | 71 (0.9%) | 602 (7.3%) |
| context_switched | 0 | 0 | 1 (0.0%) |
| shared | 0 | 0 | 0 |

**No "shared" features detected:** The shared criterion (|CDS| < 25th percentile AND mean activation > 50th percentile in both conditions) found zero features. This is because high-activation features tend to also have non-trivial CDS — features that fire consistently across 10K windows for both conditions are also the ones with the most opportunity to show condition-specific differences. Consider relaxing the shared-activation threshold in revision.

**Context-switched is near-zero:** Only 1 feature at the late layer qualifies. The Jaccard < 0.3 threshold is strict for high-expansion SAEs where sparse features activate on narrow window subsets by construction.

---

## 3. Per-Pair CDS Ranges

| Pair | Layer | CDS min | CDS max |
|---|---|---|---|
| blood (K562 vs HSC) | early | -8.70 | 14.45 |
| liver (HepG2 vs Liver) | early | -13.87 | 10.18 |
| lymph (GM12878 vs NaiveB) | early | -28.69 | 45.16 |
| blood | mid | -10.84 | 13.20 |
| liver | mid | -16.92 | 13.19 |
| lymph | mid | -12.79 | 33.44 |
| blood | late | -44.18 | 15.04 |
| liver | late | -157.93 | 9.36 |
| lymph | late | -19.28 | 108.43 |

**Liver at late layer is an outlier:** CDS min = −157.9 (vs −44.2 for blood, −19.3 for lymph). This is consistent with liver tissue having the most distinct chromatin accessibility signature of the three pairs — hepatocytes are highly differentiated primary cells vs. the HepG2 hepatocellular carcinoma line, which has substantially altered chromatin. This pair should be highlighted in the paper.

---

## 4. Permutation Test

The 10,000-permutation null (vectorized batched GPU matmul, ~1 sec total) provides clean separation between real and null CDS. Numbers above are Bonferroni-corrected at α=0.05.

Raw (uncorrected) significant fraction is substantially higher — the Bonferroni correction is conservative for 8,192 tests. BH-FDR results will be reported in Phase 7 stats.

---

## 5. Top Features Exported

Top-50 highest-|CDS| features per layer (150 total) exported to `outputs/top_features/`. Each feature directory contains:
- `activations.tsv` — per-window activation values across all 6 conditions
- `top_windows_{side}_{pair}.bed` — top 500 activating windows per condition/pair

Bio team handoff ready. HOMER + GO + ChromHMM annotation can proceed.

---

## 6. Deviations from Spec

| Item | Detail |
|---|---|
| No "shared" features | Activation + low-CDS joint criterion not satisfied. Likely needs threshold adjustment. |
| Context-switched near-zero | Jaccard < 0.3 threshold strict for sparse BatchTopK features. |
| Sign consistency low | Expected given 3-pair dataset; not a bug, a biological finding. |
