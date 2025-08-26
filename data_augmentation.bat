@echo off
rem chcp 65001 > nul
rem echo.
rem echo   ██████╗  █████╗ ██████╗ ██╗██████╗ 
rem echo   ██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗
rem echo   ██████╔╝███████║██████╔╝██║██║  ██║
rem echo   ██╔══██╗██╔══██║██╔═══╝ ██║██║  ██║
rem echo   ██║  ██║██║  ██║██║     ██║██████╔╝
rem echo   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝
rem echo.
echo  === XRD RAPID Pipeline ===
echo  Author: Suk Jin Mun (GitHub: SukjinMun)
echo          Yoonsoo Nam
echo          Sungkyun Choi
echo  Version: 1.0.0
echo.
echo  Purpose: Generates synthetic XRD datasets by
echo           parameter sweeps and refinements for
echo           CNN model training.
echo  =======================================
echo.
timeout /t 3 > nul


REM ============================
REM Step 0: Save the Original Directory
REM ============================
set "CURRENT_DIR=%cd%"

REM ============================
REM Step 1: Navigate to Batch File Directory
REM ============================
cd /d "%~dp0"

REM ============================
REM Step 2: Activate the Conda Environment
REM ============================
call conda activate py27
if errorlevel 1 (
    echo Failed to activate Conda environment 'py27'.
    pause
    exit /b 1
)

REM ============================
REM Step 3: Enable Delayed Variable Expansion
REM ============================
setlocal enabledelayedexpansion

REM ============================
REM Step 4: Generate a Unique Timestamp
REM ============================

REM Use WMIC to get a consistent timestamp in YYYYMMDD_HHMMSS format
rem for /f "tokens=2 delims==" %%I in ('"wmic os get localdatetime /value"') do set "datetime=%%I"
rem set "TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%"
for /f "usebackq tokens=* delims=" %%i in (
    `powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`
) do (
    set "TIMESTAMP=%%i"
)

echo TIMESTAMP=%TIMESTAMP%

REM ============================
REM Step 5: Create Timestamped Subfolder
REM ============================

REM Get subfolder name from inputs.txt
set "SUBFOLDER_BASE=Si"
set "found_marker=0"
for /f "usebackq delims=" %%a in (inputs.txt) do (
    if "!found_marker!"=="1" (
        set "SUBFOLDER_BASE=%%a"
        set "found_marker=2"
    )
    if "%%a"=="5>Enter the name of the subfolder to be created in the data directory (the purpose of the subfolder is to organize and store related data files and refinement results for analysis):" (
        set "found_marker=1"
    )
)
echo Using subfolder base name: !SUBFOLDER_BASE!

REM Define the subfolder name with timestamp
set "SUBFOLDER_NAME=!SUBFOLDER_BASE!_%TIMESTAMP%"

REM Create the timestamped subfolder inside 'data' directory
mkdir "data\%SUBFOLDER_NAME%"
if not exist "data\%SUBFOLDER_NAME%" (
    echo Failed to create session folder: data\%SUBFOLDER_NAME%
    pause
    exit /b 1
)

REM ============================
REM Step 6: Copy and Rename inputs.txt
REM ============================

REM Define paths
set "ORIGINAL_INPUTS=inputs.txt"
set "SESSION_INPUTS=data\%SUBFOLDER_NAME%\inputs_%TIMESTAMP%.txt"

REM Copy and rename inputs.txt to the session folder
copy "%ORIGINAL_INPUTS%" "%SESSION_INPUTS%"
if errorlevel 1 (
    echo Failed to copy %ORIGINAL_INPUTS% to %SESSION_INPUTS%
    pause
    exit /b 1
)

REM ============================
REM Step 7: Update subfolder_name.txt
REM ============================

REM Write the subfolder name to subfolder_name.txt within the session folder
echo %SUBFOLDER_NAME% > "data\%SUBFOLDER_NAME%\subfolder_name_%TIMESTAMP%.txt"

REM ============================
REM Step 8: Read Session-Specific inputs.txt
REM ============================
REM Initialize variables
set "RUN_STEPS_56="
set "FLAG=0"
REM Read the session-specific inputs.txt file line by line
for /f "usebackq tokens=* delims=" %%A in ("%SESSION_INPUTS%") do (
    set "LINE=%%A"
    REM Check if the line starts with "18>"
    if "!LINE!"=="18>Run scripts step 5 and step 6 for R-factor analysis and contour plots (Y/N)?:" (
        set "FLAG=1"
    ) else (
        if "!FLAG!"=="1" (
            REM Capture the next line after "18>"
            set "RUN_STEPS_56=!LINE!"
            set "FLAG=0"
            goto found
        )
    )
)
:found
REM Trim leading spaces and get the first token (the user's input)
for /f "tokens=1" %%C in ("%RUN_STEPS_56%") do set "RUN_STEPS_56=%%C"
REM Debugging: Print the value of RUN_STEPS_56
echo Value of RUN_STEPS_56: [%RUN_STEPS_56%]

REM ============================
REM Step 9: Run Python Scripts Based on RUN_STEPS_56
REM ============================
REM Always run steps 1-4
echo Running steps 1-4 for basic data generation.
python "scripts\step1_cif2pcr.py" "%SESSION_INPUTS%"
python "scripts\step2_pcrfolders_modifiedparameters.py" "%SESSION_INPUTS%"
python "scripts\step3_runautofp_fixall.py" "%SESSION_INPUTS%"
python "scripts\step3p5_distribution_classification.py" "%SESSION_INPUTS%"
python "scripts\step4_overplots_3datfiles.py" "%SESSION_INPUTS%"

REM Check if steps 5 and 6 should be run
if /i "%RUN_STEPS_56%"=="Y" (
    echo Running additional steps 5 and 6 for R-factor analysis.
    python "scripts\step5_parameter_1D_distribution.py" "%SESSION_INPUTS%"
    python "scripts\step6_generate_parameter_Rfactor_analysis.py" "%SESSION_INPUTS%"
) else (
    echo Skipping steps 5 and 6 as per user configuration.
)

REM ============================
REM Step 10: Cleanup - Delete subfolder_name.txt
REM ============================

REM Delete the subfolder_name.txt to prevent conflicts in future sessions
del "data\%SUBFOLDER_NAME%\subfolder_name_%TIMESTAMP%.txt"
if errorlevel 1 (
    echo Warning: Failed to delete subfolder_name.txt in data\%SUBFOLDER_NAME%
    REM Not exiting because this is a non-critical warning
)

REM ============================
REM Step 11: Switch to Base Conda Environment
REM ============================

REM Deactivate any current environment and activate base
call conda deactivate
call conda activate base


REM Change back to the original directory
cd "%CURRENT_DIR%"

rem REM Prompt the user to press any key to continue
pause