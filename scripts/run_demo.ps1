# End-to-end demo: download bottle, build memory bank, run inference.
# Run from project root with the conda env active:
#   conda activate anomalydet
#   pwsh scripts/run_demo.ps1

param(
    [string]$Category = 'bottle',
    [string]$Config   = 'configs/default.yaml'
)

$ErrorActionPreference = 'Stop'

Write-Host "==> step 1/3: download MVTec category '$Category'"
python scripts/download_mvtec.py --category $Category

Write-Host "==> step 2/3: train (build memory bank)"
python -m src.train --config $Config --category $Category

Write-Host "==> step 3/3: inference"
python -m src.inference --config $Config --category $Category `
    --memory-bank "outputs/$Category/memory_bank.pt"

Write-Host "==> done. results under outputs/$Category/predictions/"
