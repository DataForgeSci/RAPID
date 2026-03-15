@echo off
chcp 65001 > nul
echo.
echo   ██████╗  █████╗ ██████╗ ██╗██████╗ 
echo   ██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗
echo   ██████╔╝███████║██████╔╝██║██║  ██║
echo   ██╔══██╗██╔══██║██╔═══╝ ██║██║  ██║
echo   ██║  ██║██║  ██║██║     ██║██████╔╝
echo   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝
echo.
echo  === XRD Analysis + Identification Tool ===
echo.
echo  Purpose: XRD Analysis Tool for material identification
echo           using database fingerprinting and CNN models
echo           to perform automated Rietveld refinement.
echo  =======================================
echo.
timeout /t 3 > nul
setlocal enabledelayedexpansion

:: Ensure conda environment is activated
call conda activate rapid_py311 >nul 2>&1 || (
    echo Error: Could not activate Conda rapid_py311 environment
    echo Please ensure Conda is installed and initialized
    pause
    exit /b 1
)

:: Fix OpenMP library conflict between PyTorch and NumPy/SciPy
set KMP_DUPLICATE_LIB_OK=TRUE

:: Create necessary directories
if not exist "single_phase_identification" mkdir "single_phase_identification"
if not exist "single_phase_identification\database" mkdir "single_phase_identification\database"
if not exist "single_phase_identification\unknown_materials" mkdir "single_phase_identification\unknown_materials"
if not exist "single_phase_identification\identified_materials" mkdir "single_phase_identification\identified_materials"

:: Main menu
:main_menu
cls
echo.
echo =============================================
echo    XRD ANALYSIS TOOL - ENHANCED VERSION
echo =============================================
echo.
echo [1] Update/Create Material Database
echo [2] Identify Unknown Material
echo [3] Direct Model Inference
echo [4] Exit
echo.
set /p "menu_choice=Enter your choice (1-4): "

if "%menu_choice%"=="1" goto update_database
if "%menu_choice%"=="2" goto identify_material
if "%menu_choice%"=="3" goto direct_inference
if "%menu_choice%"=="4" goto exit_program
goto main_menu

:: Update database option
:update_database
cls
echo.
echo =============================================
echo    UPDATING MATERIAL DATABASE
echo =============================================
echo.
echo This will scan saved models and create fingerprints.
echo.
set /p "continue=Continue? (y/n): "
if /i not "%continue%"=="y" goto main_menu

echo.
echo Processing...
python scripts\database_manager.py --update
if %errorlevel% neq 0 (
    echo.
    echo Error: Database update failed!
    pause
) else (
    echo.
    echo Database successfully updated!
    pause
)
goto main_menu

:: Identify material option
:identify_material
cls
echo.
echo =============================================
echo    IDENTIFY UNKNOWN MATERIAL
echo =============================================
echo.

:: Check if database exists
if not exist "single_phase_identification\database\material_catalog.json" (
    echo Error: Material catalog not found.
    echo.
    echo Would you like to create the database now?
    set /p "create_db=Enter y/n: "
    if /i "%create_db%"=="y" (
        goto update_database
    ) else (
        goto main_menu
    )
)

:: List all .dat files in the unknown_materials folder
echo Available .dat files:
echo.

set "file_count=0"
for %%f in (single_phase_identification\unknown_materials\*.dat) do (
    set /a "file_count+=1"
    echo   [!file_count!] %%~nxf
)

if %file_count%==0 (
    echo No .dat files found.
    echo.
    echo Please place your unknown XRD pattern file in:
    echo single_phase_identification\unknown_materials
    echo.
    pause
    goto main_menu
)

echo.
echo [0] Return to main menu
echo.
set /p "file_choice=Select file number: "

if "%file_choice%"=="0" goto main_menu

:: Find the selected file
set "selected_file="
set "file_idx=0"
for %%f in (single_phase_identification\unknown_materials\*.dat) do (
    set /a "file_idx+=1"
    if "!file_idx!"=="%file_choice%" (
        set "selected_file=%%f"
    )
)

if not defined selected_file (
    echo Invalid selection.
    pause
    goto identify_material
)

echo.
echo Selected: !selected_file!
echo.

:: Run the identification script
echo Step 1: Identifying material and predicting parameters...
echo.

python scripts\material_identifier.py --file="!selected_file!"

if %errorlevel% neq 0 (
    echo.
    echo Error: Identification process failed.
    pause
    goto main_menu
)

:: Get the latest identified material folder
set "latest_folder="
set "latest_time=0"

for /f "tokens=*" %%d in ('dir /b /ad "single_phase_identification\identified_materials\unknown_*"') do (
    for /f "tokens=3 delims=_" %%t in ("%%d") do (
        if %%t GTR !latest_time! (
            set "latest_time=%%t"
            set "latest_folder=%%d"
        )
    )
)

if "!latest_folder!"=="" (
    echo.
    echo Error: Could not find the identified material folder.
    pause
    goto main_menu
)

set "identified_dir=single_phase_identification\identified_materials\!latest_folder!"
echo Found identified material folder: !identified_dir!

:: Step 2: Rietveld Refinement Step 1 - Create PCR file
echo.
echo Step 2: Creating PCR file with predicted parameters...
python scripts\Rietveld_Refinement_step1_MI.py "!identified_dir!"

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to create PCR file.
    pause
    goto main_menu
)

:: Step 3: Rietveld Refinement Step 2 - Run AutoFP (in py27 environment)
echo.
echo Step 3: Running Rietveld refinement with AutoFP (using Python 2.7)...

:: Save current environment
set "PREV_ENV="
for /f "tokens=*" %%i in ('conda info --envs ^| findstr "*"') do (
    for %%j in (%%i) do set "PREV_ENV=%%j"
)

:: Switch to py27 for step 2
call conda activate rapid_py27
python scripts\Rietveld_Refinement_step2_MI.py "!identified_dir!"
set refinement_status=%errorlevel%

:: Switch back to previous environment
call conda activate !PREV_ENV!

if %refinement_status% neq 0 (
    echo.
    echo Error: Rietveld refinement failed.
    pause
    goto main_menu
)

:: Step 4: Rietveld Refinement Step 3 - Analysis and plots
echo.
echo Step 4: Analyzing refinement results and creating plots...
python scripts\Rietveld_Refinement_step3_MI.py "!identified_dir!"

if %errorlevel% neq 0 (
    echo.
    echo Error: Refinement analysis failed.
    pause
    goto main_menu
)

:: Complete - Show summary and open folder
echo.
echo =============================================
echo    MATERIAL IDENTIFICATION COMPLETE
echo =============================================
echo.
echo All refinement steps completed successfully!
echo Results are saved in:
echo !identified_dir!
echo.

rem :: Ask if user wants to open the folder
rem set /p "open_folder=Open results folder? (y/n): "
rem if /i "!open_folder!"=="y" (
rem     start explorer "!identified_dir!"
rem )

pause
goto main_menu

:direct_inference
cls
echo.
echo =============================================
echo    DIRECT MODEL INFERENCE
echo =============================================
echo.
if not exist "single_phase_identification\database\material_catalog.json" (
    echo Error: Material catalog not found.
    echo.
    echo Would you like to create the database now?
    set /p "create_db=Enter y/n: "
    if /i "%create_db%"=="y" (
        goto update_database
    ) else (
        goto main_menu
    )
)
echo Available .dat files:
echo.
set "file_count=0"
for %%f in (single_phase_identification\unknown_materials\*.dat) do (
    set /a "file_count+=1"
    echo   [!file_count!] %%~nxf
)
if %file_count%==0 (
    echo No .dat files found.
    echo.
    echo Please place your XRD pattern file in:
    echo single_phase_identification\unknown_materials
    echo.
    pause
    goto main_menu
)
echo.
echo [0] Return to main menu
echo.
set /p "file_choice=Select file number: "
if "%file_choice%"=="0" goto main_menu
set "selected_file="
set "file_idx=0"
for %%f in (single_phase_identification\unknown_materials\*.dat) do (
    set /a "file_idx+=1"
    if "!file_idx!"=="%file_choice%" (
        set "selected_file=%%f"
    )
)
if not defined selected_file (
    echo Invalid selection.
    pause
    goto direct_inference
)
echo.
echo Selected: !selected_file!
echo.
echo Available materials: Ba, CeO2, pbso4, tbbaco
set /p "material_name=Enter material name: "
if "%material_name%"=="" (
    echo Error: Material name cannot be empty.
    pause
    goto direct_inference
)
echo.
echo Running direct inference with %material_name% model...
echo.
python scripts\direct_inference.py --file="!selected_file!" --material="%material_name%"
if %errorlevel% neq 0 (
    echo.
    echo Error: Direct inference failed.
    pause
    goto main_menu
)
set "latest_folder="
set "latest_time=0"
for /f "tokens=*" %%d in ('dir /b /ad "single_phase_identification\identified_materials\unknown_*"') do (
    for /f "tokens=3,4 delims=_" %%t in ("%%d") do (
        set "combined=%%t%%u"
        if !combined! GTR !latest_time! (
            set "latest_time=!combined!"
            set "latest_folder=%%d"
        )
    )
)
if not defined latest_folder (
    echo Warning: Could not find results folder.
    pause
    goto main_menu
)
echo.
echo =============================================
echo Step 2: Creating PCR file with predicted parameters...
python scripts\Rietveld_Refinement_step1_inference.py "single_phase_identification\identified_materials\!latest_folder!"
if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to create PCR file.
    pause
    goto main_menu
)
echo.
echo Step 3: Running Rietveld refinement with AutoFP (using Python 2.7)...
set "PREV_ENV="
for /f "tokens=*" %%i in ('conda info --envs ^| findstr "*"') do (
    for %%j in (%%i) do set "PREV_ENV=%%j"
)
call conda activate rapid_py27
python scripts\Rietveld_Refinement_step2_inference.py "single_phase_identification\identified_materials\!latest_folder!"
set refinement_status=%errorlevel%
call conda activate !PREV_ENV!
if %refinement_status% neq 0 (
    echo.
    echo Error: Rietveld refinement failed.
    pause
    goto main_menu
)
echo.
echo Step 4: Analyzing refinement results and creating plots...
python scripts\Rietveld_Refinement_step3_inference.py "single_phase_identification\identified_materials\!latest_folder!"
if %errorlevel% neq 0 (
    echo.
    echo Error: Refinement analysis failed.
    pause
    goto main_menu
)
echo.
echo =============================================
echo Direct inference completed successfully!
echo Results saved to: single_phase_identification\identified_materials\!latest_folder!
echo =============================================
echo.
pause
goto main_menu

:exit_program
echo.
echo Thank you for using the XRD Analysis Tool.
exit /b 0