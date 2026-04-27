#!/usr/bin/env Rscript
# Thin wrapper: delegates GO enrichment to Python (g:Profiler REST API).
# No Bioconductor packages needed.

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

top_features_dir <- parse_arg("--top_features_dir", "/workspace/outputs/top_features")
annotation_dir   <- parse_arg("--annotation_dir",   "/workspace/outputs/annotation/go")
layers_str       <- parse_arg("--layers",            "early mid late")
n_top            <- parse_arg("--n_top",             "50")
genome           <- parse_arg("--genome",            "hg38")

py_script <- "/workspace/src/go_enrichment.py"

cmd <- paste(
  "python3", shQuote(py_script),
  "--top_features_dir", shQuote(top_features_dir),
  "--annotation_dir",   shQuote(annotation_dir),
  "--layers",           shQuote(layers_str),
  "--n_top",            n_top,
  "--genome",           genome
)

cat("Delegating GO enrichment to Python:\n ", cmd, "\n")
exit_code <- system(cmd)
if (exit_code != 0) {
  cat("[WARN] GO enrichment returned non-zero exit code:", exit_code, "\n")
  cat("Continuing — GO results will be missing but pipeline will not fail.\n")
}
