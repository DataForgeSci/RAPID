@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo    MODEL ORGANIZATION UTILITY
echo ===================================================
echo.

cd ..\saved_models
if errorlevel 1 (
    echo Error: Could not navigate to saved_models directory.
    pause
    exit /b 1
)

echo Current directory: %cd%
echo.

if not exist "backup" (
    echo Creating backup folder...
    mkdir "backup"
)

echo Looking for models_* folders...
set "models_folder_found=0"

for /d %%M in (models_*) do (
    set "models_folder=%%M"
    set "models_folder_found=1"
    
    echo.
    echo Found models folder: !models_folder!
    echo.
    
    echo Moving model folders into !models_folder!...
    set "model_count=0"
    
    for /d %%F in (model_*) do (
        echo   Moving %%F into !models_folder!...
        move "%%F" "!models_folder!\" >nul
        if errorlevel 1 (
            echo   ERROR: Failed to move %%F
        ) else (
            set /a "model_count+=1"
            echo   Moved successfully.
        )
    )
    
    echo Moved !model_count! model folders into !models_folder!
    
    echo Moving !models_folder! to backup folder...
    move "!models_folder!" "backup\" >nul
    if errorlevel 1 (
        echo ERROR: Failed to move !models_folder! to backup folder.
    ) else (
        echo Successfully moved !models_folder! to backup folder.
    )
)

if !models_folder_found! EQU 0 (
    echo No models_* folders found.
)

echo.
echo Organization completed.
echo ===================================================
echo.
pause