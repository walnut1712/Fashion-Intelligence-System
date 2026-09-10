$ErrorActionPreference = "Stop"

Write-Host "=== 1/2 T1 120x160 background adaptation ==="
& .\.venv\Scripts\python.exe .\scripts\train_t12_background_adaptation.py --task 1 --epochs 8

Write-Host ""
Write-Host "=== 2/2 T4 + T1 semantic reranking evaluation ==="
& .\.venv\Scripts\python.exe .\scripts\evaluate_t1_t4_semantic_rerank.py --tune-queries 500 --test-queries 1000 --pool 100

Write-Host ""
Write-Host "DONE. Read:"
Write-Host "  outputs\evaluation\task1_bgadapt_comparison.csv"
Write-Host "  outputs\evaluation\task1_t4_rerank_test.csv"
Write-Host "  outputs\evaluation\task1_t4_rerank_summary.json"
