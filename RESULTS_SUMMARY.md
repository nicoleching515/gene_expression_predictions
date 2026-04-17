# RESULTS_SUMMARY.md

## 1. Context Divergence Score (Phase 4)

| Layer | Bonferroni-significant features | % of 8,192 | CDS std |
|---|---|---|---|
| early (L/4) | 57  | 0.70% | 0.195 |
| mid   (L/2) | 82  | 1.00% | 0.229 |
| late (3L/4) | 215 | 2.62% | 0.657 |

Signal grows 3.7× from early to late layer. Liver pair shows strongest single-pair divergence (max |CDS| = 157.9 at late layer).

## 2. Feature Categories (mid layer, Phase 4)

| Category | Count | % |
|---|---|---|
| other | 8,007 | 97.7% |
| vitro_enriched | 114 | 1.4% |
| vivo_enriched | 71 | 0.9% |

Vitro-enriched consistently outnumber vivo-enriched (~1.5×) across all layers.

## 3. Ablation (Phase 5, mid layer)

| k | Targeted Δŷ | Random Δŷ (mean) | Fold over random |
|---|---|---|---|
| 5   | 4.97 | 4.45 | 1.12× |
| 10  | 8.11 | 4.48 | 1.81× |
| **25** | **12.36** | **4.53** | **2.73×** |
| 50  | 12.74 | 4.80 | 2.65× |
| 100 | 13.60 | 5.19 | 2.62× |

**Primary causal test (all k × seeds, n=25 pairs):**
- Wilcoxon p = **2.98×10⁻⁸**
- Cohen's d = **1.789** (very large effect)

k=25 is the optimal ablation depth (highest fold-enrichment over random).

## 4. Steering Gap Closure (Phase 6, mid layer)

| Setting | Median GC | 95% CI | Frac > 0.5 |
|---|---|---|---|
| Best: α=2.0, β=0.5 | **0.112** | [0.000, 0.171] | 29.5% |
| Fixed: α=2.0, β=0.25 | 0.041 | [0.000, 0.138] | 28.5% |
| Random steering | −0.025 | — | — |
| Direct context swap | 1.000 | — | 100% |

SAE steering explains ~11% of the vivo–vitro prediction gap; 4.5× above random steering.

## 5. Cross-layer Feature Overlap (Phase 7, Fig 7)

| Pair | Jaccard | p (hypergeometric) | BH-FDR p |
|---|---|---|---|
| early ∩ mid | 0.020 | 0.037 | 0.074 |
| early ∩ late | 0.000 | 1.000 | 1.000 |
| mid ∩ late | 0.000 | 1.000 | 1.000 |

Top-50 context-divergent features are nearly entirely non-overlapping across layers — evidence for hierarchical, non-redundant chromatin representation in EpiBERT.

## 6. Pending (requires Bio1 annotation)

- Fig 3: feature annotation heatmap (HOMER + GO + ChromHMM)
- Fig 6: case studies of top divergent features
- Linear probe correction baseline for steering
- BH-FDR analysis of CDS (expect more features than Bonferroni)
