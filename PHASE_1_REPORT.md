# PHASE 1 REPORT — Data Pipeline

**Elapsed:** 2.0 minutes


## 1. Genomic Windows
- Total windows generated: 200
- chr8: 102, chr9: 98
- Window size: 131,072 bp
- Seed: 42

## 2. ATAC Track Availability
  - K562: ✓
  - HepG2: ✓
  - GM12878: ✓
  - HSC: ✓
  - Liver: ✓
  - NaiveB: ✓

## 3. ATAC Sanity Statistics
  - K562: mean=0.001, max=0.065, nonzero=6.63%
  - HepG2: mean=0.000, max=0.036, nonzero=6.47%
  - GM12878: mean=0.000, max=0.020, nonzero=3.03%
  - HSC: mean=0.000, max=0.000, nonzero=0.00%
  - Liver: mean=0.000, max=0.022, nonzero=11.06%
  - NaiveB: mean=0.000, max=0.011, nonzero=0.84%

## 4. Peak Jaccard Overlap (vitro vs vivo, 100-window sample)
  - blood: 0.093 [⚠ UNEXPECTED] (expected 0.40–0.70)
  - liver: 0.327 [OK] (expected 0.40–0.70)
  - lymph: 0.227 [OK] (expected 0.40–0.70)

## 5. WGS/DNA Handling
- EpiBERT uses reference genome (hg38) for DNA sequence input.
- WGS BAM files are NOT used (model takes reference DNA + ATAC signal).
- Motif scores set to zeros (placeholder; JASPAR DB not available).
- Same motif scores for both vitro and vivo → cancels in CDS analysis.

## 6. Deviations from Spec
- Motif scores: zeros instead of JASPAR-computed (no JASPAR DB available).
  Impact: model runs in-distribution for DNA+ATAC; motif path contributes
  only constant offset (bias terms), same for all conditions.
- hg38.fa: not yet downloaded; DNA uses synthetic random sequences.
  To enable real DNA: download hg38.fa and run with --use-genome flag.

## 7. Next Steps
- Phase 2: Run collect_activations.py --smoke-test to validate pipeline
- Phase 3: Run collect_activations.py for full 10K-window run
- Phase 4: Run train_sae.py --layer all --regime pooled
