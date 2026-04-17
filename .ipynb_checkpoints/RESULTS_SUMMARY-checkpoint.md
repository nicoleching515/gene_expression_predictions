# RESULTS_SUMMARY.md

## 1. Context Divergence Score

- Layer early: 0.70% features significant (Bonferroni α=0.05)
- Layer mid: 1.00% features significant (Bonferroni α=0.05)
- Layer late: 2.62% features significant (Bonferroni α=0.05)

## 2. Feature Categories (mid layer)

- other: 8007 (97.7%)
- vitro_enriched: 114 (1.4%)
- vivo_enriched: 71 (0.9%)

## 3. Ablation (k=25, mid layer)

- Targeted Δŷ: 12.3609 ± 8.7520
- Random Δŷ:   4.5269 ± 6.3684
- Wilcoxon p-value: 1
- Cohen's d: 52.878

## 4. Steering Gap Closure

- Best (α=2.0, β=0.5): median GC = 0.112
- Fixed (α=2.0, β=0.25): median GC = 0.041 [0.000, 0.138]
- Frac windows GC > 0.5: 28.50%

## 5. Cross-layer Feature Overlap

- early_vs_mid: Jaccard=0.020, p=0.03713
- early_vs_late: Jaccard=0.000, p=1
- mid_vs_late: Jaccard=0.000, p=1
