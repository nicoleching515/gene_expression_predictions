# PHASE 5 REPORT — Ablation Experiments

**Target layer:** mid
**Eval windows:** 200
**k sweep:** [5, 10, 25, 50, 100]
**Random seeds:** 5

## Results
      k   ablation_type  delta_mean  delta_std  seed
0     5        targeted    4.973871   6.250962    -1
1     5  top_activation    6.731421   7.226808    -1
2     5          random    4.607878   6.803487     0
3     5          random    4.350842   6.088052     1
4     5          random    4.408998   6.150516     2
5     5          random    4.309435   6.037160     3
6     5          random    4.579955   6.794092     4
7    10        targeted    8.114069   8.919888    -1
8    10  top_activation   22.712365  17.638315    -1
9    10          random    4.573202   6.405962     0
10   10          random    4.468632   6.151025     1
11   10          random    4.515901   6.297123     2
12   10          random    4.362930   6.162300     3
13   10          random    4.469527   6.270680     4
14   25        targeted   12.360929   8.752041    -1
15   25  top_activation   26.794999  18.112558    -1
16   25          random    4.863622   6.901296     0
17   25          random    4.552432   6.498598     1
18   25          random    4.606508   6.347764     2
19   25          random    4.334151   5.918996     3
20   25          random    4.277960   6.175198     4
21   50        targeted   12.741883   9.924917    -1
22   50  top_activation   14.624129  16.427039    -1
23   50          random    4.722949   6.685695     0
24   50          random    4.599473   6.231594     1
25   50          random    5.218552   7.750585     2
26   50          random    4.918414   6.927796     3
27   50          random    4.545734   6.159369     4
28  100        targeted   13.602705   9.886558    -1
29  100  top_activation   28.139593   8.399655    -1
30  100          random    5.216321   7.196202     0
31  100          random    4.785761   6.075647     1
32  100          random    4.855268   6.386852     2
33  100          random    5.490347   7.873552     3
34  100          random    5.608550   8.548210     4
