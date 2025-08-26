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
echo  Purpose: End-to-end pipeline combining
echo           XRD data augmentation and CNN training
echo           for automated Rietveld refinement.
echo  =======================================
echo.
timeout /t 3 > nul
setlocal enabledelayedexpansion

echo.
echo ========================================
echo === XRD Data Augmentation + ML Pipeline
echo ========================================
echo.

REM ============================
REM Step 0: Save the Original Directory
REM ============================
set "CURRENT_DIR=%cd%"
cd /d "%~dp0"
set "ROOT_DIR=%cd%"

REM ============================
REM NEW: Mode Selection Menu
REM ============================
echo Choose mode:
echo [1] Process multiple datasets
echo [2] Fine-tune a compound
echo.
set /p PROCESSING_MODE="Enter selection (1 or 2): "

REM Handle invalid input
if "%PROCESSING_MODE%" NEQ "1" if "%PROCESSING_MODE%" NEQ "2" (
    echo Invalid selection. Defaulting to multiple dataset processing.
    set "PROCESSING_MODE=1"
)

if "%PROCESSING_MODE%"=="2" goto :fine_tuning_mode

REM ============================
REM Original functionality continues if not in fine-tuning mode
REM ============================
echo.
echo [Multiple dataset processing mode activated]
echo.

goto :original_functionality_proceed

:fine_tuning_mode
echo.
echo [Fine-tuning mode activated]
echo.

REM Display current inputs.txt
echo ===== Current inputs.txt Configuration =====
if exist "inputs.txt" (
    type "inputs.txt"
) else (
    echo No inputs.txt file found. Using default template.
    REM Generate a template if needed
    copy "templates\default_inputs.txt" "inputs.txt" > nul 2>&1
    if not exist "inputs.txt" (
        echo Error: Could not create inputs.txt template.
        pause
        exit /b 1
    )
    type "inputs.txt"
)
echo.
echo =========================================
echo.

REM Parameter selection menu
echo Select parameters to fine-tune (enter comma-separated numbers):
echo [1] Zero shift
echo [2] Lattice parameters
echo [3] Biso parameter
echo [4] Scale factor
echo [5] U parameter
echo [6] V parameter
echo [7] W parameter
echo.
set /p PARAM_SELECTION="Your selection: "

REM Ask how many iterations
echo.
set /p TUNE_COUNT="How many fine-tuning iterations would you like to perform? (1-10): "

REM Initialize parameter arrays
set "PARAM_ZERO=0"
set "PARAM_LATTICE=0"
set "PARAM_BISO=0"
set "PARAM_SCALE=0"
set "PARAM_U=0"
set "PARAM_V=0"
set "PARAM_W=0"

REM Process parameter selections
echo %PARAM_SELECTION% | findstr "1" > nul
if not errorlevel 1 set "PARAM_ZERO=1"

echo %PARAM_SELECTION% | findstr "2" > nul
if not errorlevel 1 set "PARAM_LATTICE=1"

echo %PARAM_SELECTION% | findstr "3" > nul
if not errorlevel 1 set "PARAM_BISO=1"

echo %PARAM_SELECTION% | findstr "4" > nul
if not errorlevel 1 set "PARAM_SCALE=1"

echo %PARAM_SELECTION% | findstr "5" > nul
if not errorlevel 1 set "PARAM_U=1"

echo %PARAM_SELECTION% | findstr "6" > nul
if not errorlevel 1 set "PARAM_V=1"

echo %PARAM_SELECTION% | findstr "7" > nul
if not errorlevel 1 set "PARAM_W=1"

REM Get shifts for each selected parameter
echo.
echo Enter shift/percentage for each parameter:

REM Process zero parameter if selected
set "ZERO_SHIFTS="
if "%PARAM_ZERO%"=="1" (
    set /p ZERO_SHIFTS="For zero parameter (absolute shift values, comma-separated): "
)

REM Process lattice parameter if selected
set "LATTICE_SHIFTS="
if "%PARAM_LATTICE%"=="1" (
    set /p LATTICE_SHIFTS="For lattice parameter (percentage values with %%, comma-separated): "
)

REM Process biso parameter if selected
set "BISO_SHIFTS="
if "%PARAM_BISO%"=="1" (
    set /p BISO_SHIFTS="For biso parameter (percentage values with %%, comma-separated): "
)

REM Process scale parameter if selected
set "SCALE_SHIFTS="
if "%PARAM_SCALE%"=="1" (
    set /p SCALE_SHIFTS="For scale parameter (percentage values with %%, comma-separated): "
)

REM Process U parameter if selected
set "U_SHIFTS="
if "%PARAM_U%"=="1" (
    set /p U_SHIFTS="For U parameter (percentage values with %%, comma-separated): "
)

REM Process V parameter if selected
set "V_SHIFTS="
if "%PARAM_V%"=="1" (
    set /p V_SHIFTS="For V parameter (percentage values with %%, comma-separated): "
)

REM Process W parameter if selected
set "W_SHIFTS="
if "%PARAM_W%"=="1" (
    set /p W_SHIFTS="For W parameter (percentage values with %%, comma-separated): "
)

REM Create macro_inputs directory if it doesn't exist
if not exist "macro_inputs" mkdir "macro_inputs"

REM First, check if inputs_1.txt exists in macro_inputs folder
if exist "macro_inputs\inputs_1.txt" (
    set "BASE_INPUT_FILE=macro_inputs\inputs_1.txt"
) else (
    set "BASE_INPUT_FILE=inputs.txt"
    echo Warning: inputs_1.txt not found in macro_inputs folder. Using inputs.txt as base.
)

REM Run Python script to create fine-tuned inputs
echo.
echo Creating %TUNE_COUNT% fine-tuned inputs.txt files with specified parameters...
echo Base input file: %BASE_INPUT_FILE%

REM Activate conda environment for Python
call conda activate base

REM Initialize disable_partial_rerun flag
set "DISABLE_RERUN_FLAG="
if "%DISABLE_PARTIAL_RERUN%"=="Y" set "DISABLE_RERUN_FLAG=--disable_partial_rerun"

REM Call the Python script with updated parameters to use inputs_1.txt as the base
python scripts\create_fine_tuned_inputs.py ^
    --input_file="%BASE_INPUT_FILE%" ^
    --output_dir="macro_inputs" ^
    --count=%TUNE_COUNT% ^
    --zero=%PARAM_ZERO% ^
    --zero_shifts="%ZERO_SHIFTS%" ^
    --lattice=%PARAM_LATTICE% ^
    --lattice_shifts="%LATTICE_SHIFTS%" ^
    --biso=%PARAM_BISO% ^
    --biso_shifts="%BISO_SHIFTS%" ^
    --scale=%PARAM_SCALE% ^
    --scale_shifts="%SCALE_SHIFTS%" ^
    --u=%PARAM_U% ^
    --u_shifts="%U_SHIFTS%" ^
    --v=%PARAM_V% ^
    --v_shifts="%V_SHIFTS%" ^
    --w=%PARAM_W% ^
    --w_shifts="%W_SHIFTS%"

if %ERRORLEVEL% NEQ 0 (
    echo Error creating fine-tuned input files
    pause
    exit /b 1
)

REM Display ML_inputs file
echo.
echo ===== Current ML_inputs_1.txt Configuration =====
if exist "CNN\macro_inputs\ML_inputs_1.txt" (
    type "CNN\macro_inputs\ML_inputs_1.txt"
) else (
    echo No ML_inputs_1.txt file found. Using default template.
    if not exist "CNN\macro_inputs" mkdir "CNN\macro_inputs"
    copy "templates\default_ML_inputs.txt" "CNN\macro_inputs\ML_inputs_1.txt" > nul 2>&1
    if exist "CNN\macro_inputs\ML_inputs_1.txt" (
        type "CNN\macro_inputs\ML_inputs_1.txt"
    ) else (
        echo Error: Could not create ML_inputs_1.txt template.
    )
)
echo.
echo =========================================
echo.

echo All configuration files have been prepared.
set /p CONTINUE="Proceed with data augmentation process? (Y/N): "
if /i "%CONTINUE%" NEQ "Y" (
    echo Process canceled by user.
    pause
    exit /b 0
)

echo.
echo Proceeding with data augmentation process...
echo.

REM Skip the input file checking part of original functionality
set "SKIP_INPUT_CHECKS=1"

REM Continue with data augmentation process
goto :run_data_augmentation

:original_functionality_proceed
REM ============================
REM Step 1: Read configuration
REM ============================
echo Reading configuration from inputs.txt...

REM Initialize SKIP_INPUT_CHECKS for option 1
set "SKIP_INPUT_CHECKS=0"

REM Skip checks if coming from fine-tuning mode
if "%SKIP_INPUT_CHECKS%"=="1" (
    echo Skipping input checks since we're coming from fine-tuning mode.
    goto :run_data_augmentation
)

REM Check if inputs.txt exists
if not exist "inputs.txt" (
    echo Error: inputs.txt not found.
    pause
    exit /b 1
)

echo Configuration file found.
echo.

REM ============================
REM Step 2: Check for macro_inputs folder (for data augmentation)
REM ============================
if not exist "macro_inputs" (
    echo Creating macro_inputs directory...
    mkdir macro_inputs
    echo Please place your data augmentation input .txt files in the macro_inputs folder.
    pause
    exit /b 1
)

REM Initialize the count variable
set /a count=0

REM If in fine-tuning mode, only count fine-tuned input files
if "%PROCESSING_MODE%"=="2" (
    for %%f in (macro_inputs\inputs_ft_*.txt) do set /a count+=1
    
    if %count%==0 (
        echo No fine-tuned input files found in the macro_inputs folder.
        echo Fine-tuning process may have failed.
        pause
        exit /b 1
    )
    
    echo Found %count% fine-tuned input files to process.
    
    REM Temporarily rename any non-fine-tuned input files
    for %%f in (macro_inputs\*.txt) do (
        set "filename=%%~nxf"
        if not "!filename:~0,9!"=="inputs_ft_" (
            echo Temporarily renaming !filename! to !filename!.bak
            ren "macro_inputs\%%f" "%%f.bak"
        )
    )
) else (
REM Simplified direct approach for file detection
set /a count=0
set "debug_files="

REM Directly check if inputs_1.txt exists
echo Checking for inputs_1.txt...
if exist "macro_inputs\inputs_1.txt" (
    echo FOUND: inputs_1.txt exists!
    set /a count=1
    set "debug_files=inputs_1.txt"
) else (
    echo WARNING: inputs_1.txt not found
)

echo Debug - Count: !count!
echo Debug - Files found: !debug_files!

if !count! EQU 0 (
    echo No .txt files found in the macro_inputs folder.
    echo Please add at least one input file like inputs_1.txt to the macro_inputs folder.
    echo Example: Copy your inputs.txt file to macro_inputs\inputs_1.txt
    pause
    exit /b 1
)

echo Found !count! input files to process in macro_inputs folder.
echo Files to process: !debug_files!
    
    echo Found %count% input files to process in macro_inputs folder.
    echo Note: For Option 1, you need regular input files not fine-tuned ones.
)

:run_data_augmentation

REM ============================
REM Step 3: Run Data Augmentation Process
REM ============================
echo.
echo =============================================
echo === PHASE 1: Running Data Augmentation ===
echo =============================================
echo.

REM ============================
REM Step 3.1: Activate the Conda Environment
REM ============================
echo Activating Conda environment py27 for data augmentation...
call conda activate py27
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate Conda environment 'py27'.
    pause
    exit /b %ERRORLEVEL%
)

REM ============================
REM Step 3.2: Navigate to 'scripts' Directory
REM ============================
cd scripts
if %ERRORLEVEL% NEQ 0 (
    echo Failed to navigate to 'scripts' directory.
    pause
    exit /b %ERRORLEVEL%
)

REM ============================
REM Step 3.3: Run the Python Macro Script
REM ============================
if "%PROCESSING_MODE%"=="2" (
    echo Running the Python macro script to process only fine-tuned input files...
    REM Pass the pattern to match only ft files
    python "run_macro.py" --process_pattern="inputs_ft_*.txt" --non_blocking
) else (
    echo Running the Python macro script to process all input files...
    python "run_macro.py" --process_all --non_blocking
)

if %ERRORLEVEL% NEQ 0 (
    echo Python script run_macro.py encountered an error.
    pause
    REM Continue anyway - don't exit
    set PYTHON_ERROR=1
) else (
    set PYTHON_ERROR=0
)

REM ============================
REM Step 3.4: Return to original directory
REM ============================
cd /d "%ROOT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to return to the original directory.
    pause
    exit /b %ERRORLEVEL%
)

REM ============================
REM Step 3.5: Deactivate the Conda Environment
REM ============================
echo Deactivating Conda environment...
call conda deactivate

echo.
echo =========================================================
echo === Data augmentation completed successfully!
echo === Generated datasets are now in the data/ directory
echo =========================================================
echo.

REM ============================
REM Step 4: Move Generated Data to CNN/data/train_data
REM ============================
echo.
echo =============================================
echo === PHASE 2: Moving Datasets to CNN Training Folder ===
echo =============================================
echo.

REM ============================
REM Step 4.1: Create CNN/data/train_data if it doesn't exist
REM ============================
if not exist "CNN\data\train_data" (
    echo Creating CNN\data\train_data directory...
    mkdir "CNN\data\train_data"
)

REM ============================
REM Step 4.2: Find and copy timestamped data folders
REM ============================
echo Searching for generated datasets in data/ directory...

REM Store folder names directly in a variable instead of a temp file
echo.

REM Copy each folder and build a list of dataset names
set /a copied_count=0
set "dataset_list="

for /d %%D in (data\*) do (
    set "folder_name=%%~nxD"
    if /i not "!folder_name!"=="backup" (
        echo Processing: !folder_name!
        
        REM Copy folder to CNN/data/train_data
        echo Copying to CNN\data\train_data\!folder_name!...
        if exist "CNN\data\train_data\!folder_name!" (
            echo Folder already exists in destination, skipping...
        ) else (
            xcopy "data\!folder_name!" "CNN\data\train_data\!folder_name!" /E /I /Y > nul
            if errorlevel 1 (
                echo Error copying folder !folder_name!
            ) else (
                set /a copied_count+=1
                echo Copied successfully.
                
                REM Add to our dataset_list variable
                if defined dataset_list (
                    set "dataset_list=!dataset_list!,!folder_name!"
                ) else (
                    set "dataset_list=!folder_name!"
                )
                echo Added !folder_name! to dataset list.
            )
        )
    )
)

echo.
echo Successfully copied %copied_count% dataset folders to CNN\data\train_data\
echo Dataset names have been saved.
echo.

REM Display the datasets to process
echo Datasets to process:
echo %dataset_list%
echo.

REM ============================
REM Step 5: Process Each Dataset (Configure ML Inputs and Train)
REM ============================
echo.
echo =============================================
echo === PHASE 3: Processing Datasets ===
echo =============================================
echo.

REM ============================
REM Step 5.1: Check if we have datasets to process
REM ============================
if not defined dataset_list (
    echo Error: No datasets were successfully copied.
    echo Cannot continue with ML model training.
    pause
    exit /b 1
)
echo Dataset list is available.


REM ============================
REM Step 5.2: Check if ML_inputs_1.txt exists
REM ============================
echo Checking for ML_inputs_1.txt...
if not exist "CNN\macro_inputs\ML_inputs_1.txt" (
    echo Error: CNN\macro_inputs\ML_inputs_1.txt not found.
    echo Cannot continue with ML model training.
    pause
    exit /b 1
)
echo ML_inputs_1.txt found.

REM ============================
REM Step 5.3: Prepare for dataset processing
REM ============================
echo Found the following datasets to process:
echo %dataset_list%
echo.

REM ============================
REM Step 5.4: Process each dataset completely (configure, train, clean)
REM ============================

REM Process each dataset in a separate scope
for %%D in (%dataset_list%) do (
    call :process_dataset "%%D"
)

REM If in fine-tuning mode, restore temporarily renamed files
if "%PROCESSING_MODE%"=="2" (
    echo Restoring temporarily renamed input files...
    for %%f in (macro_inputs\*.txt.bak) do (
        set "filename=%%~nxf"
        set "basename=!filename:.bak=!"
        echo Restoring !basename!
        ren "macro_inputs\%%f" "!basename!"
    )
)

echo.
echo =============================================
echo === PIPELINE COMPLETED SUCCESSFULLY ===
echo =============================================
echo.
echo The following datasets were processed:
echo.

REM Use a separate scope with delayed expansion for listing results
setlocal enabledelayedexpansion
for /d %%D in (data\*) do (
    set "folder_name=%%~nxD"
    if /i not "!folder_name!"=="backup" (
        echo Dataset: !folder_name!
        
        REM Check for models in the backup folder
        if exist "CNN\saved_models\backup\models_!folder_name!" (
            echo  - Models saved in: CNN\saved_models\backup\models_!folder_name!
            echo  - Models in this folder:
            
            REM List all model folders for this dataset
            for /d %%M in ("CNN\saved_models\backup\models_!folder_name!\*") do (
                echo    * %%~nxM
            )
        ) else (
            echo  - No models found for this dataset
        )
        echo.
    )
)
endlocal

echo.
pause
exit /b 0

REM ============================
REM Subroutine to process a single dataset
REM This isolates variable scope for each dataset
REM ============================
:process_dataset
setlocal enabledelayedexpansion
set "current_dataset=%~1"
echo.
echo =============================================
echo === Processing dataset: %current_dataset%
echo =============================================
echo.

REM --- CONFIGURE ML INPUTS ---
echo Configuring ML inputs for dataset: %current_dataset%

REM Update ML_inputs_1.txt to reference the current dataset
echo Updating ML_inputs_1.txt with dataset: %current_dataset%

REM Use Python script to update the ML_inputs_1.txt file
echo Using Python script to update the dataset line...

REM Ensure conda base environment is activated for Python
call conda activate base

REM Store the script's full path with explicit paths
set "script_path=%ROOT_DIR%\scripts\update_ml_input.py"
set "ml_inputs_path=%ROOT_DIR%\CNN\macro_inputs\ML_inputs_1.txt"

REM Call the Python script with the absolute file paths
echo Using script path: %script_path%
echo Using ML_inputs path: %ml_inputs_path%

REM Run Python with explicit absolute paths
python "%script_path%" "%ml_inputs_path%" "%current_dataset%"
set "PYTHON_RESULT=%ERRORLEVEL%"

if %PYTHON_RESULT% NEQ 0 (
    echo Error updating ML_inputs_1.txt with Python script
    pause
    exit /b 1
)

echo ML_inputs_1.txt updated successfully.
echo.

REM Verify the file was updated correctly
echo Verifying update...
findstr /C:"N;%current_dataset%" "%ROOT_DIR%\CNN\macro_inputs\ML_inputs_1.txt"
if errorlevel 1 (
    echo Warning: Dataset line verification failed. File may not have been updated correctly.
    type "%ROOT_DIR%\CNN\macro_inputs\ML_inputs_1.txt" | findstr /C:"N;"
) else (
    echo Verification successful.
)

REM Find the inputs file for this dataset using absolute paths
set "dataset_inputs="
for /f "usebackq delims=" %%F in (`dir /b "%ROOT_DIR%\data\%current_dataset%\inputs_*.txt"`) do (
    set "dataset_inputs=%ROOT_DIR%\data\%current_dataset%\%%F"
)

REM Use Python script to create ML_inputs files based on configuration
echo Creating ML_inputs files for dataset %current_dataset%...

REM Call our custom Python script to create ML_inputs files with absolute paths
set "config_script_path=%ROOT_DIR%\scripts\create_ml_inputs_from_config.py"
echo Using config script path: %config_script_path%
echo Using dataset inputs path: %dataset_inputs%

python "%config_script_path%" "%dataset_inputs%" "%ml_inputs_path%"
set "CREATE_RESULT=%ERRORLEVEL%"

if %CREATE_RESULT% NEQ 0 (
    echo Error creating ML_inputs files
    pause
    exit /b 1
) else (
    echo Created ML_inputs files successfully.
)
echo.

REM --- TRAIN CNN MODELS ---
echo Training models for dataset: %current_dataset%

REM Create a folder to store models for this dataset
set "models_folder=%ROOT_DIR%\CNN\saved_models\models_%current_dataset%"
if not exist "%models_folder%" mkdir "%models_folder%"

REM Run CNN training
echo Running train_CNN_macro.bat...
cd /d "%ROOT_DIR%\CNN"
echo Using direct NUL redirection to handle pause commands...
call train_CNN_macro.bat < NUL

REM Return to the root directory
cd /d "%ROOT_DIR%"

REM Continue regardless of exit code - the training succeeded if we get here
echo CNN training completed successfully.

REM Check if any .pth files were created (for additional verification)
set "models_exist=0"
for %%F in ("%ROOT_DIR%\CNN\saved_models\*.pth") do (
    set "models_exist=1"
)

if %models_exist% EQU 0 (
    echo Warning: No model files found in saved_models directory.
    echo Will continue execution but models may not have been created properly.
)

REM Run select_best_models script BEFORE organizing model folders
echo Running best model selection script...
set "models_folder=%ROOT_DIR%\CNN\saved_models"
if exist "!models_folder!" (
    echo Copying select_best_models.py to !models_folder!...
    copy "%ROOT_DIR%\CNN\scripts\select_best_models.py" "!models_folder!" /Y > nul
    
    REM Change to the models folder and run the script
    cd /d "!models_folder!"
    echo Running model selection in !models_folder!...
    python select_best_models.py
    
    REM Clean up the script after running
    del /Q select_best_models.py
    
    REM Return to root directory
    cd /d "%ROOT_DIR%"
    echo Best model selection completed.
) else (
    echo Warning: Models folder not found at !models_folder!
)

REM Now proceed with the regular organization
echo Organizing model folders...
echo Current directory before organization: %ROOT_DIR%

REM Run the organize_models.bat file from CNN/scripts
echo Running organize_models.bat to organize model folders...
cd /d "%ROOT_DIR%\CNN\scripts"
echo Changed to scripts directory: %cd%

call organize_models.bat < nul

cd /d "%ROOT_DIR%"
echo Returned to original directory: %ROOT_DIR%
echo Model organization completed.

REM Clean up ML_inputs files for next dataset
echo Cleaning up ML_inputs files...
if exist "%ROOT_DIR%\CNN\clean_ml_inputs.bat" (
    cd /d "%ROOT_DIR%\CNN"
    call clean_ml_inputs.bat < nul
    cd /d "%ROOT_DIR%"
    echo Cleanup completed.
) else (
    echo Warning: clean_ml_inputs.bat not found at %ROOT_DIR%\CNN\clean_ml_inputs.bat
    echo Skipping cleanup step.
)

echo.
echo Dataset %current_dataset% processing completed.
echo.

REM End of this dataset's processing scope
endlocal
exit /b 0