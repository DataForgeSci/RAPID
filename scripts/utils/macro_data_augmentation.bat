@echo off



set "CURRENT_DIR=%cd%"


cd /d "%~dp0"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to navigate to the Batch file directory.
    pause
    exit /b %ERRORLEVEL%
)


if not exist "macro_inputs" (
    echo Creating macro_inputs directory...
    mkdir macro_inputs
    echo Please place your input .txt files in the macro_inputs folder.
    pause
    exit /b 1
)

set /a count=0
for %%f in (macro_inputs\*.txt) do set /a count+=1

if %count%==0 (
    echo No .txt files found in the macro_inputs folder.
    echo Please add your input files and run this batch file again.
    pause
    exit /b 1
)


CALL conda activate py27
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate Conda environment 'py27'.
    pause
    exit /b %ERRORLEVEL%
)


cd scripts
if %ERRORLEVEL% NEQ 0 (
    echo Failed to navigate to 'scripts' directory.
    pause
    exit /b %ERRORLEVEL%
)


echo Running the Python macro script to process all input files...
python "run_macro.py" --process_all --non_blocking
if %ERRORLEVEL% NEQ 0 (
    echo Python script run_macro.py encountered an error.
    pause
    exit /b %ERRORLEVEL%
)


CALL conda deactivate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to deactivate Conda environment.
    pause
    exit /b %ERRORLEVEL%
)


cd /d "%CURRENT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to return to the original directory.
    pause
    exit /b %ERRORLEVEL%
)


echo.
echo ======================================================
echo All sessions have been processed successfully.
echo Press any key to exit...
echo ======================================================
pause