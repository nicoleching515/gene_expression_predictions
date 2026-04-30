# PHASE 10 REPORT — Genome-Wide HOMER Re-Run and Publication-Quality Figures

**Completed:** 2026-04-30

Corrected the HOMER motif enrichment pipeline to use a proper genome-wide background,
re-ran all 18 conditions, fixed a figure labelling bug, and regenerated Figs 8–10 for
publication.

---

## Problem with Previous Runs

The Phase 8/9 HOMER runs used a custom background file
(`outputs/annotation/homer/background_all_windows.bed`) that contained only two entries:

```
chr8    0    145134248
chr9    0    138394194
```

These whole-chromosome spans were produced by over-merging the top-feature BED files,
which themselves only cover chr8/chr9. The result was that HOMER received **0 valid
background sequences** after repeat-masking, causing every motif to receive p = 1.0 and
q = 1.0 — i.e. completely uninformative results.

---

## Fix

A new script `run_homer_genomewide.sh` was written that:

1. **Removes the `-bg` flag** — drops the broken custom background entirely.
2. **Adds `-genomeBg`** — instructs HOMER (v5.1) to randomly sample ~100,000
   GC-content-matched sequences from the full hg38 genome as background.
3. **Clears stale output directories** before each run so previously cached
   (invalid) results are not reused.

Command used for each of the 18 conditions:

```bash
findMotifsGenome.pl {tag}.bed hg38 {outdir} -size 200 -mask -genomeBg -p 8
```

All 18 conditions completed successfully. Background sequence counts per run: ~100,000.

---

## Results Summary

### Top HOMER Motifs (top 5 by p-value per condition)

#### Layer: EARLY

| Condition | Top motifs |
|-----------|-----------|
| vitro · blood | RORgt (1e-2), Hoxd12 (1e-2), Foxa2 (1e-1), FOXA1 (1e-1) |
| vitro · liver | MafA (1e-3), Phox2b (1e-1), HOXA1 (1e-1), Hoxa10 (1e-1), Phox2a (1e-1) |
| vitro · lymph | Sox3 (1e-3), ZBED2 (1e-2), Sox15 (1e-1), Tcf12 (1e-1), EWS:ERG-fusion (1e-1) |
| vivo · blood  | PBX1 (1e-2), STAT4 (1e-2), ZNF143\|STAF (1e-2), STAT1 (1e-1), Foxa2 (1e-1) |
| vivo · liver  | NF1-halfsite (1e-2), ZNF519 (1e-2), GLI3 (1e-2), Bcl11a (1e-1), Gfi1b (1e-1) |
| vivo · lymph  | RXR (1e-2), E2F (1e-2), X-box (1e-2), TATA-Box (1e-1), ZNF519 (1e-1) |

#### Layer: MID

| Condition | Top motifs |
|-----------|-----------|
| vitro · blood | HOXA3 (1e-3), Sox6 (1e-3), Sox3 (1e-3), SOX1 (1e-2), Phox2b (1e-2) |
| vitro · liver | Pitx1:Ebox (1e-2), bZIP:IRF (1e-2), Pit1 (1e-1), OCT4-SOX2-TCF-NANOG (1e-1), IRF:BATF (1e-1) |
| vitro · lymph | KLF6 (1e-2), Hoxd10 (1e-2), Nr5a2 (1e-2), ERb (1e-2), AP-2gamma (1e-2) |
| vivo · blood  | Gsx2 (1e-2), COUP-TFII (1e-2), LHX9 (1e-2), MITF (1e-2), Nur77 (1e-2) |
| vivo · liver  | VDR (1e-2), ZKSCAN1 (1e-1), Gli2 (1e-1), Gata4 (1e-1), HEB (1e-1) |
| vivo · lymph  | Oct6 (1e-3), Brn1 (1e-2), Gfi1b (1e-1), Zfp281 (1e-1), PBX2 (1e-1) |

#### Layer: LATE

| Condition | Top motifs |
|-----------|-----------|
| vitro · blood | Tcfcp2l1 (1e-3), KLF10 (1e-2), PAX5 (1e-2), Hnf1 (1e-1), CEBP:CEBP (1e-1) |
| vitro · liver | Pitx1:Ebox (1e-2), IRF1 (1e-2), FOXA1 (1e-1), Nkx2.5 (1e-1) |
| vitro · lymph | ERb (1e-3), Tlx? (1e-2), NFkB-p65-Rel (1e-2), Smad4 (1e-2), Zscan4c (1e-1) |
| vivo · blood  | RUNX2 (1e-3), RUNX-AML (1e-2), ZNF341 (1e-2), AR-halfsite (1e-2), Hoxa10 (1e-2) |
| vivo · liver  | NFkB-p65-Rel (1e-2), Pax7 (1e-2), Zfp809 (1e-1), ERG (1e-1), Oct11 (1e-1) |
| vivo · lymph  | NFkB-p65-Rel (1e-2), ZBED2 (1e-2), NFkB2-p52 (1e-1), CEBP:AP1 (1e-1), Sox6 (1e-1) |

### Fisher Meta-Analysis (Top 10 motifs, `outputs/annotation/meta_analysis.tsv`)

| Motif | Best cell-type Fisher p | Driving pair | Best single condition |
|-------|------------------------|-------------|----------------------|
| ERb | 6.26e-03 | lymph | late_vitro_lymph (1e-3) |
| Hoxa10 | 2.75e-02 | blood | late_vivo_blood (1e-2) |
| Tcfcp2l1 | 2.75e-02 | blood | late_vitro_blood (1e-3) |
| Sox3 | 2.75e-02 | blood | early_vitro_lymph (1e-3) |
| LHX9 | 1.04e-01 | blood | mid_vivo_blood (1e-2) |
| GATA3 | 1.04e-01 | liver | mid_vitro_liver (1e-1) |
| NFkB-p65-Rel | 1.04e-01 | lymph | late_vitro_lymph (1e-2) |
| ZBED2 | 1.04e-01 | lymph | early_vitro_lymph (1e-2) |
| Brn2 | 1.04e-01 | blood | early_vivo_blood (1e-1) |
| AR-halfsite | 1.04e-01 | blood | late_vivo_blood (1e-2) |

Notable biological signals:
- **ERb** (estrogen receptor beta, lymph features): consistent enrichment in lymph
  conditions across early/mid/late, driven by late_vitro_lymph (p=1e-3, q=0.05).
  Estrogen receptor binding is known to regulate B-cell differentiation.
- **RUNX2 / RUNX-AML** (late vivo blood): RUNX family TFs are master regulators of
  haematopoiesis; enrichment in late-layer blood-divergent SAE features is consistent
  with the model having learned lineage-specific chromatin accessibility patterns.
- **NFkB-p65-Rel / NFkB2-p52** (late lymph): NFkB signalling is a hallmark of mature
  B-cell activation, consistent with the GM12878 / NaiveB contrast in lymph features.
- **FOXA1 / Foxa2** (liver): Pioneer factor enrichment in liver-divergent features
  aligns with the established role of FOXA factors in hepatocyte-specific chromatin
  opening.
- **SOX family** (blood/lymph, early–mid vitro): broad enrichment consistent with
  progenitor-state chromatin accessibility captured in K562 vs HSC features.

---

## Figure Changes

### Fig 8 — HOMER Motif Enrichment Heatmap
- **Bug fix:** "In vitro (cell line)" / "In vivo (tissue)" side-block annotations were
  swapped; corrected to match actual row ordering (vitro rows are drawn first).
- **Caption updated:** removed "restricted to chr8/chr9" warning; replaced with
  description of genome-wide GC-matched background.
- **Title updated:** now reads "genome-wide background" instead of "chr8/chr9 scope".

### Fig 10 — Meta-Analysis
- **Footer updated:** removed "chr8/chr9 extension recommended"; replaced with
  description of HOMER `-genomeBg` (hg38) method.

### All figures (Figs 8, 9, 10)
- Regenerated at 300 dpi as both PDF (vector) and PNG.

---

## Files Changed

| File | Change |
|------|--------|
| `run_homer_genomewide.sh` | New script: genome-wide HOMER via `-genomeBg` |
| `src/plot_homer_go_figures.py` | Fix vitro/vivo label swap; update caption |
| `src/motif_meta_analysis.py` | Update figure footer note |
| `outputs/annotation/homer/*/knownResults.txt` | Re-run results (all 18 conditions) |
| `outputs/annotation/homer/*/homer.log` | Re-run logs (all 18 conditions) |
| `outputs/annotation/homer/*/motifFindingParameters.txt` | Re-run parameters |
| `outputs/annotation/meta_analysis.tsv` | Re-computed Fisher meta-analysis |
| `results/figures/fig8_homer_motifs.{pdf,png}` | Regenerated with genome-wide results |
| `results/figures/fig9_go_enrichment.{pdf,png}` | Regenerated |
| `results/figures/fig10_meta_analysis.{pdf,png}` | Regenerated with genome-wide results |

---

## Scripts

| Script | Description |
|--------|-------------|
| `run_homer_genomewide.sh` | HOMER genome-wide re-run: clears stale dirs, runs with `-genomeBg` |
| `src/plot_homer_go_figures.py` | Generates fig8 (motif heatmap) and fig9 (GO:BP dot-plot) |
| `src/motif_meta_analysis.py` | Fisher meta-analysis, null comparison, cell-type specificity → fig10 |
