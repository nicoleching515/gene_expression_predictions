#!/usr/bin/env python3
"""
scripts/copy_models_to_workspace.py
====================================
Download (or symlink from HF cache) the three external DNA foundation
models to /workspace/gene_expression_predictions/models/ so they can be
loaded from a local path rather than requiring network access.

Models:
  - LongSafari/hyenadna-medium-160k-seqlen-hf   → models/hyenadna-medium-160k
  - InstaDeepAI/nucleotide-transformer-v2-500m-multi-species → models/nucleotide-transformer-v2-500m
  - EleutherAI/enformer-official-rough           → models/enformer-official-rough

Usage:
    cd /workspace/gene_expression_predictions
    python scripts/copy_models_to_workspace.py

The script uses huggingface_hub.snapshot_download, which will reuse
already-cached files via hard-links and won't re-download anything.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("LongSafari/hyenadna-medium-160k-seqlen-hf",
     "hyenadna-medium-160k"),
    ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
     "nucleotide-transformer-v2-500m"),
    ("EleutherAI/enformer-official-rough",
     "enformer-official-rough"),
]

for repo_id, subdir in MODELS:
    dest = MODELS_DIR / subdir
    print(f"\n=== {repo_id} → models/{subdir} ===")
    if dest.exists() and any(dest.iterdir()):
        print(f"  Already present at {dest}, skipping.")
        continue
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,   # copy, not symlink, for portability
        )
        print(f"  Done. Saved to {dest}")
    except Exception as e:
        print(f"  ERROR downloading {repo_id}: {e}")

print("\nAll models processed. Check models/ directory.")
