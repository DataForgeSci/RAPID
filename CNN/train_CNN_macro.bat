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
echo  Purpose: Trains CNN models on XRD datasets and
echo           performs Rietveld Refinement with the
echo           predicted parameters.
echo  =======================================
echo.
timeout /t 3 > nul
setlocal enabledelayedexpansion

:: Activate conda environment
call conda activate base || (
    echo Error: Could not activate Conda base environment
    echo Please ensure Conda is installed and initialized
    pause
    exit /b 1
)

:: Directory containing the multiple ML_inputs*.txt files
set "MACRO_DIR=macro_inputs"

:: Check if macro_inputs directory exists
if not exist "%MACRO_DIR%" (
    echo Error: %MACRO_DIR% directory not found
    pause
    exit /b 1
)

:: Initialize SELECTED
set "SELECTED="

:menu
cls
echo Files to be processed:
echo -------------------------------

:: Build a list of all files and automatically add to SELECTED
set "SELECTED="
set "file_num=0"
set "total_files=0"

:: Count total files first
for %%f in ("%MACRO_DIR%\ML_inputs*.txt") do (
    set /a total_files+=1
)

:: If no ML_inputs found, exit
if %total_files%==0 (
    echo No ML_inputs*.txt files found in %MACRO_DIR%
    pause
    exit /b 1
)

rem :: Display all files and add to SELECTED
rem for %%f in ("%MACRO_DIR%\ML_inputs*.txt") do (
rem     set /a "file_num+=1"
rem     echo [!file_num!] %%~nxf
rem     set "SELECTED=!SELECTED!%%~nxf;"
rem )

rem echo.
rem echo Found %total_files% configuration files in %MACRO_DIR% folder.
rem echo.
rem set /p "confirm=Process all these files? (y/n): "
rem if /i not "%confirm%"=="y" exit /b 0

rem goto :run_sessions

rem :: Show content of selected file and confirm
rem cls
rem set "selected_file=!file_%choice%!"
rem echo Content of !selected_file!:
rem echo -------------------------------
rem type "%MACRO_DIR%\!selected_file!"
rem echo -------------------------------
rem echo.

rem set /p "confirm=Use this config? (y/n): "
rem if /i "%confirm%"=="y" (
rem     :: add to SELECTED
rem     echo !SELECTED! | findstr /i /c:"!selected_file!" >nul || (
rem         set "SELECTED=!SELECTED!!selected_file!;"
rem     )
rem )

rem :: ask if more
rem echo.
rem set /p "add_more=Add another configuration to macro processing? (y/n): "
rem if /i "%add_more%"=="y" goto :menu

rem :run_sessions
rem cls
rem if "%SELECTED%"=="" (
rem     echo No configurations selected
rem     pause
rem     goto :menu
rem )

rem echo Selected configurations:
rem for %%a in ("%SELECTED:;=" "%") do (
rem     if not "%%~a"=="" echo  %%~a
rem )
rem echo.
rem set /p "final_confirm=Commence macro with these sessions? (y/n): "
rem if /i not "%final_confirm%"=="y" goto :menu

:: Display all files and automatically select them
for %%f in ("%MACRO_DIR%\ML_inputs*.txt") do (
    set /a "file_num+=1"
    echo [!file_num!] %%~nxf
    set "SELECTED=!SELECTED!%%~nxf;"
)

echo.
echo Found %total_files% configuration files in %MACRO_DIR% folder.
echo All files will be processed automatically.
echo.

goto :run_sessions

:run_sessions
if "%SELECTED%"=="" (
    echo No configurations found to process
    exit /b 1
)

echo Processing the following configurations:
for %%a in ("%SELECTED:;=" "%") do (
    if not "%%~a"=="" echo  %%~a
)
echo.

:: Now process each file in SELECTED
for %%a in ("%SELECTED:;=" "%") do (
    if not "%%~a"=="" (
        echo Processing: %%~a
        echo -------------------
        :: 1) Train CNN
        python scripts\train_xrd.py "%MACRO_DIR%\%%~a"
        echo Training completed for %%~a

        :: 2) Rietveld refinement step 1 - the script checks line5
        echo Now calling Rietveld_Refinement_step1.py with same file
        python scripts\Rietveld_Refinement_step1.py "%MACRO_DIR%\%%~a"
        
        :: 3) Run Rietveld refinement step 2 in py27 environment
        echo.
        echo Running Rietveld refinement step 2
        :: Save current environment
        set "PREV_ENV="
        for /f "tokens=*" %%i in ('conda info --envs ^| findstr "*"') do (
            for %%j in (%%i) do set "PREV_ENV=%%j"
        )
        :: Switch to py27 for step 2
        call conda activate py27
        python scripts\Rietveld_Refinement_step2.py "%MACRO_DIR%\%%~a"
        :: Switch back to previous environment
        call conda activate !PREV_ENV!
        
        :: 4) Run Rietveld refinement step 3 for analysis
        echo.
        echo Running Rietveld refinement step 3
        python scripts\Rietveld_Refinement_step3.py "%MACRO_DIR%\%%~a"
        echo Refinement analysis completed
        
        echo -------------------
        echo.
    )
)
echo All training sessions completed!
pause
exit /b 0