@echo off
REM Run Rwp Uncertainty for ALL materials (CeO2, pbso4, tbbaco) - ALL 10,000 patterns each
REM Just double-click to run!

setlocal

set START_IDX=0
set END_IDX=9999
set SCRIPT_DIR=%~dp0

REM Initialize conda for cmd.exe
call C:\Users\Qiuyu\anaconda3\Scripts\activate.bat C:\Users\Qiuyu\anaconda3

echo ############################################################
echo #                                                          #
echo #   RWP UNCERTAINTY QUANTIFICATION - ALL MATERIALS         #
echo #   Processing ALL 10,000 patterns per material            #
echo #                                                          #
echo ############################################################
echo.
echo Patterns to process: %START_IDX% to %END_IDX% (10,000 patterns)
echo Materials: CeO2, pbso4, tbbaco
echo.

REM ============================================================
REM MATERIAL 1: CeO2
REM ============================================================
echo.
echo ############################################################
echo #  MATERIAL 1/3: CeO2                                      #
echo ############################################################
call "%SCRIPT_DIR%run_rwp_uncertainty.bat" CeO2 %START_IDX% %END_IDX%
if errorlevel 1 (
    echo WARNING: CeO2 processing had errors
)

REM ============================================================
REM MATERIAL 2: pbso4
REM ============================================================
echo.
echo ############################################################
echo #  MATERIAL 2/3: pbso4                                     #
echo ############################################################
call "%SCRIPT_DIR%run_rwp_uncertainty.bat" pbso4 %START_IDX% %END_IDX%
if errorlevel 1 (
    echo WARNING: pbso4 processing had errors
)

REM ============================================================
REM MATERIAL 3: tbbaco
REM ============================================================
echo.
echo ############################################################
echo #  MATERIAL 3/3: tbbaco                                    #
echo ############################################################
call "%SCRIPT_DIR%run_rwp_uncertainty.bat" tbbaco %START_IDX% %END_IDX%
if errorlevel 1 (
    echo WARNING: tbbaco processing had errors
)

echo.
echo ############################################################
echo #                                                          #
echo #   ALL MATERIALS COMPLETE                                 #
echo #                                                          #
echo ############################################################
echo.
echo Results saved in each model's backup folder under:
echo   saved_models/backup/[model_folder]/uncertainty_results/rwp_uncertainty/
echo.
echo Summary files:
echo   - CeO2: rwp_summary_%START_IDX%_to_%END_IDX%.dat
echo   - pbso4: rwp_summary_%START_IDX%_to_%END_IDX%.dat
echo   - tbbaco: rwp_summary_%START_IDX%_to_%END_IDX%.dat
echo.
echo Press any key to exit...
pause >nul

endlocal
