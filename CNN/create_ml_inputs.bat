@echo off
setlocal enabledelayedexpansion

:: Directory where ML_inputs files will be created
set "MACRO_DIR=macro_inputs"

:: Check if original ML_inputs_1.txt exists
if not exist "%MACRO_DIR%\ML_inputs_1.txt" (
    echo Error: Original template file %MACRO_DIR%\ML_inputs_1.txt not found
    echo Please ensure this file exists before running this script
    pause
    exit /b 1
)

:: Ask for the total number of files to have
set /p "total_files=How many total ML_inputs_x.txt files do you want in the %MACRO_DIR% folder? "

:: Validate input is a number
echo %total_files%| findstr /r "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo Error: Please enter a valid positive number
    pause
    exit /b 1
)

:: If user wants fewer than 2 files, no need to create additional files
if %total_files% LSS 2 (
    echo No additional files needed. %MACRO_DIR%\ML_inputs_1.txt already exists.
    pause
    exit /b 0
)

:: Read through the file to find the model line
set "found_model=0"
set "base_model="
for /f "usebackq tokens=*" %%a in ("%MACRO_DIR%\ML_inputs_1.txt") do (
    set "line=%%a"
    if "!line:~0,2!"=="y;" (
        set "found_model=1"
        for /f "tokens=1,* delims=;" %%b in ("!line!") do (
            set "model_name=%%c"
        )
    )
)

if "!found_model!"=="0" (
    echo Error: Could not find a line starting with "y;" in %MACRO_DIR%\ML_inputs_1.txt
    echo Make sure the file has a line where the model name is specified.
    pause
    exit /b 1
)

:: Extract base model name (remove _1 if present)
set "base_model=!model_name!"
if "!model_name:~-2!"=="_1" (
    set "base_model=!model_name:~0,-2!"
)

echo Original model name: !model_name!
echo Base model name: !base_model!
echo.
echo Will create !total_files! total ML_inputs_x.txt files with model names:
echo - !base_model!_1 (already exists)
for /l %%i in (2,1,%total_files%) do (
    echo - !base_model!_%%i
)

:: Confirm before proceeding
echo.
set /p "confirm=Continue with file creation? (y/n): "
if /i not "!confirm!"=="y" (
    echo Operation cancelled by user
    pause
    exit /b 0
)

:: Create the additional files
echo.
echo Creating files...
for /l %%i in (2,1,%total_files%) do (
    set "output_file=%MACRO_DIR%\ML_inputs_%%i.txt"
    
    if exist "!output_file!" (
        echo File !output_file! already exists, updating...
    ) else (
        echo Creating new file !output_file!...
    )
    
    (
        for /f "usebackq tokens=*" %%j in ("%MACRO_DIR%\ML_inputs_1.txt") do (
            set "line=%%j"
            if "!line:~0,2!"=="y;" (
                echo y;!base_model!_%%i
            ) else (
                echo %%j
            )
        )
    ) > "!output_file!"
)

echo.
echo Successfully created %total_files% input files in the %MACRO_DIR% folder
pause
exit /b 0