@echo off
REM Run Rwp Uncertainty (Step 1 + Step 2) for a single material
REM Usage: run_rwp_uncertainty.bat <material> <start_idx> <end_idx>
REM Example: run_rwp_uncertainty.bat CeO2 0 99

setlocal

set MATERIAL=%1
set START_IDX=%2
set END_IDX=%3
set SCRIPT_DIR=%~dp0

REM Activate conda base environment for Python 3
call conda activate rapid_py311

if "%MATERIAL%"=="" (
    echo Usage: run_rwp_uncertainty.bat ^<material^> ^<start_idx^> ^<end_idx^>
    echo Available materials: CeO2, pbso4, tbbaco
    exit /b 1
)

if "%START_IDX%"=="" (
    echo Usage: run_rwp_uncertainty.bat ^<material^> ^<start_idx^> ^<end_idx^>
    exit /b 1
)

if "%END_IDX%"=="" set END_IDX=%START_IDX%

echo ============================================================
echo MATERIAL: %MATERIAL%
echo PATTERNS: %START_IDX% to %END_IDX%
echo ============================================================

echo.
echo ============================================================
echo STEP 1: CNN Inference (Python 3)
echo ============================================================

python "%SCRIPT_DIR%rwp_uncertainty_step1.py" %MATERIAL% %START_IDX% %END_IDX%

if errorlevel 1 (
    echo Step 1 failed for %MATERIAL%
    exit /b 1
)

echo.
echo ============================================================
echo STEP 2: Rietveld Refinement (Python 2.7 via conda rapid_py27)
echo ============================================================

call conda activate rapid_py27
python "%SCRIPT_DIR%rwp_uncertainty_step2.py" %MATERIAL% %START_IDX% %END_IDX%
call conda deactivate

echo.
echo ============================================================
echo COMPLETE: %MATERIAL% patterns %START_IDX% to %END_IDX%
echo ============================================================

endlocal
