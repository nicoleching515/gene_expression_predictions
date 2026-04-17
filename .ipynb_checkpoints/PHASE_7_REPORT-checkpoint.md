# PHASE 7 REPORT — Figures and Statistics

## Figures Generated
- fig2
- fig4
- fig5
- fig7
- fig3_feature_annotation
- fig6_case_studies

## Statistical Tests
                          test          layer              metric         value      pval   pval_bh
0       permutation_bonferroni          early    frac_significant  6.958008e-03       NaN       NaN
1       permutation_bonferroni            mid    frac_significant  1.000977e-02       NaN       NaN
2       permutation_bonferroni           late    frac_significant  2.624512e-02       NaN       NaN
3  wilcoxon_targeted_vs_random            mid            k25_pval  5.287755e+01  1.000000  1.000000
4              bootstrap_gc_ci            mid  median_gap_closure  4.100505e-02       NaN       NaN
5              bootstrap_gc_ci            mid               ci_lo  2.384186e-07       NaN       NaN
6              bootstrap_gc_ci            mid               ci_hi  1.382829e-01       NaN       NaN
7   hypergeometric_cross_layer   early_vs_mid     overlap_jaccard  2.040816e-02  0.037132  0.148528
8   hypergeometric_cross_layer  early_vs_late     overlap_jaccard  0.000000e+00  1.000000  1.000000
9   hypergeometric_cross_layer    mid_vs_late     overlap_jaccard  0.000000e+00  1.000000  1.000000