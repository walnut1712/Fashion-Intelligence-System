<#
.SYNOPSIS
Run the Task 1 adaptation and Task 4 semantic reranking experiment.
.DESCRIPTION
Fine-tune Task 1 with mixed backgrounds, then compare Task 4 cosine retrieval
against reranking with the original and adapted Task 1 classifiers. Evaluate on
clean catalogue queries and held-out photographic backgrounds. The Task 4
encoder stays fixed, and results are written under the selected experiment tag.
.EXAMPLE
.\scripts\run_task1_task4_reranking_experiment.ps1 -DryRun
.EXAMPLE
.\scripts\run_task1_task4_reranking_experiment.ps1 -Python .\.venv\Scripts\python.exe -Tag _rerank_experiment
#>
[CmdletBinding()]
param(
    [string]$Python,
    [ValidateRange(1, 1000)][int]$Epochs = 8,
    [ValidateRange(1, 100000)][int]$TuneQueries = 500,
    [ValidateRange(1, 100000)][int]$TestQueries = 1000,
    [ValidateRange(10, 100000)][int]$Pool = 100,
    [ValidatePattern('^_[A-Za-z0-9][A-Za-z0-9_-]*$')][string]$Tag = '_rerank_experiment',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
}
$pythonCommand = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python executable not found: $Python. Create .venv or pass -Python with an installed interpreter."
}
$pythonExecutable = $pythonCommand.Source
$trainingScript = Join-Path $PSScriptRoot 'train_task1_task2_background_adaptation.py'
$evaluationScript = Join-Path $PSScriptRoot 'eval_task1_task4_semantic_rerank.py'
$adaptedCheckpoint = Join-Path $repositoryRoot "artifacts\task1_bgaug$Tag\task1_bgadapt_best.pt"

function Invoke-PythonStep {
    param([string]$Title, [string[]]$Arguments)
    Write-Host $Title
    if ($DryRun) {
        $displayArguments = $Arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }
        Write-Host ("& '" + $pythonExecutable.Replace("'", "''") + "' " + ($displayArguments -join ' '))
        return
    }
    & $pythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE."
    }
}

Push-Location -LiteralPath $repositoryRoot
try {
    Invoke-PythonStep -Title '=== 1/2 Task 1 120x160 background adaptation ===' -Arguments @(
        $trainingScript, '--task', '1', '--epochs', "$Epochs", '--tag', $Tag
    )
    Invoke-PythonStep -Title '=== 2/2 Task 4 + Task 1 semantic reranking evaluation ===' -Arguments @(
        $evaluationScript, '--tune-queries', "$TuneQueries", '--test-queries', "$TestQueries",
        '--pool', "$Pool", '--task1-adapted', $adaptedCheckpoint, '--tag', $Tag
    )
    if ($DryRun) {
        Write-Host 'Dry run complete. No training or evaluation was started.'
    } else {
        Write-Host 'DONE. Read:'
        Write-Host "  outputs\evaluation\task1_bgadapt_comparison$Tag.csv"
        Write-Host "  outputs\evaluation\task1_t4_rerank_test$Tag.csv"
        Write-Host "  outputs\evaluation\task1_t4_rerank_summary$Tag.json"
    }
} finally {
    Pop-Location
}
