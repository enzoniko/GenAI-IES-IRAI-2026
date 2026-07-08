# Full Experiment Run Script (Overnight) with Logging
# Run from: D:\WORK\GenAI\GenAI-IES-IRAI-2026
# Usage: .\run_experiments.ps1
# Output logged to: results/run_experiments_YYYY-MM-DD_HH-MM-SS.log

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$LOG_FILE = "results\run_experiments_$timestamp.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logLine = "[$ts] [$Level] $Message"
    Add-Content -Path $LOG_FILE -Value $logLine
    switch ($Level) {
        "ERROR" { Write-Host $logLine -ForegroundColor Red }
        "WARN"  { Write-Host $logLine -ForegroundColor Yellow }
        "OK"    { Write-Host $logLine -ForegroundColor Green }
        default { Write-Host $logLine }
    }
}

# Start logging
Write-Log "========================================" "INFO"
Write-Log "IRAI 2026 Full Experiment Run" "INFO"
Write-Log "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
Write-Log "Log file: $LOG_FILE" "INFO"
Write-Log "========================================" "INFO"
Write-Log "" "INFO"

# Ensure results directory exists
if (!(Test-Path "results")) { 
    New-Item -ItemType Directory -Path "results" | Out-Null 
    Write-Log "Created results/ directory" "OK"
}
if (!(Test-Path "assets")) { 
    New-Item -ItemType Directory -Path "assets" | Out-Null 
    Write-Log "Created assets/ directory" "OK"
}

# Check existing checkpoints
Write-Log "Checking existing checkpoints..." "INFO"
$checkpoints = @("pinn.pth", "ts_jepa.pth", "decoder1.pth", "decoder2.pth", "ldm.pth")
foreach ($ckpt in $checkpoints) {
    $path = "results\$ckpt"
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1KB
        Write-Log "  Found: $ckpt ($([math]::Round($size))KB)" "OK"
    } else {
        Write-Log "  Missing: $ckpt (will train)" "WARN"
    }
}

# === PHASE 0: PINN Training ===
Write-Log "" "INFO"
Write-Log "=== PHASE 0: PINN Training ===" "INFO"
Write-Log "Command: python main_phase0.py --epochs 250 --batch_size 8" "INFO"
$phase0_start = Get-Date
python main_phase0.py --epochs 250 --batch_size 8 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
$phase0_duration = (Get-Date) - $phase0_start
if ($LASTEXITCODE -ne 0) { 
    Write-Log "Phase 0 exit code: $LASTEXITCODE" "WARN" 
} else {
    Write-Log "Phase 0 completed in $([math]::Round($phase0_duration.TotalMinutes)) min" "OK"
}

# === PHASE 1: TS-JEPA + Decoders ===
Write-Log "" "INFO"
Write-Log "=== PHASE 1: TS-JEPA + Decoders ===" "INFO"
Write-Log "Command: python main_phase1.py --max_epochs 50 --batch_size 32 --num_samples 1500" "INFO"
$phase1_start = Get-Date
python main_phase1.py --max_epochs 50 --batch_size 32 --num_samples 1500 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
$phase1_duration = (Get-Date) - $phase1_start
if ($LASTEXITCODE -ne 0) { 
    Write-Log "Phase 1 exit code: $LASTEXITCODE" "WARN" 
} else {
    Write-Log "Phase 1 completed in $([math]::Round($phase1_duration.TotalMinutes)) min" "OK"
}

# === PHASE 2: LDM Training + SDEdit Generation ===
Write-Log "" "INFO"
Write-Log "=== PHASE 2: LDM + SDEdit ===" "INFO"
Write-Log "Command: python main_phase2.py --ldm_epochs 50" "INFO"
$phase2_start = Get-Date
python main_phase2.py --ldm_epochs 50 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
$phase2_duration = (Get-Date) - $phase2_start
if ($LASTEXITCODE -ne 0) { 
    Write-Log "Phase 2 exit code: $LASTEXITCODE" "WARN" 
} else {
    Write-Log "Phase 2 completed in $([math]::Round($phase2_duration.TotalMinutes)) min" "OK"
}

# === Baseline Experiments ===
Write-Log "" "INFO"
Write-Log "=== BASELINES: Vanilla + LabelConditioned DDPM ===" "INFO"
Write-Log "Command: python train_baselines_task15.py" "INFO"
$baseline_start = Get-Date
python train_baselines_task15.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
$baseline_duration = (Get-Date) - $baseline_start
if ($LASTEXITCODE -ne 0) { 
    Write-Log "Baselines exit code: $LASTEXITCODE" "WARN" 
} else {
    Write-Log "Baselines completed in $([math]::Round($baseline_duration.TotalMinutes)) min" "OK"
}

# === TSTR Transfer Classification ===
Write-Log "" "INFO"
Write-Log "=== TSTR: Transfer Classification ===" "INFO"
Write-Log "Command: python scripts/task17_tstr_experiment.py" "INFO"
$tstr_start = Get-Date
python scripts/task17_tstr_experiment.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
$tstr_duration = (Get-Date) - $tstr_start
if ($LASTEXITCODE -ne 0) { 
    Write-Log "TSTR exit code: $LASTEXITCODE" "WARN" 
} else {
    Write-Log "TSTR completed in $([math]::Round($tstr_duration.TotalMinutes)) min" "OK"
}

# === Generate Paper Figures ===
Write-Log "" "INFO"
Write-Log "=== FIGURES: Paper-ready plots ===" "INFO"
python scripts/t10_pinn_variant_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
python scripts/task14_phase1_validation.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
Write-Log "Figures generated" "OK"

# Final summary
Write-Log "" "INFO"
Write-Log "========================================" "INFO"
$total_duration = ($phase0_start - (Get-Date))  # Will be negative, but we'll calculate manually
$end_time = Get-Date
$total_mins = (($phase0_start - $end_time).Duration() * -1).TotalMinutes
Write-Log "All experiments complete!" "OK"
Write-Log "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
Write-Log "Total runtime: see log timestamps for each phase" "INFO"
Write-Log "Log saved to: $LOG_FILE" "INFO"
Write-Log "========================================" "INFO"