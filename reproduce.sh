#!/bin/bash
# Reproduce all figures and tables from cached activations + SAE checkpoints.
# Run from /workspace/project/
set -e

cd /workspace/project

echo "=== Regenerating figures and tables ==="
python src/run_pipeline.py --start-phase 4
echo "=== Done. Results in results/ ==="
