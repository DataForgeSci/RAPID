@echo off
setlocal enabledelayedexpansion

:: Directory where ML_inputs files are stored
set "MACRO_DIR=macro_inputs"

:: Check if the directory exists
if not exist "%MACRO_DIR%" (
    echo Error: %MACRO_DIR% directory not found
    pause
    exit /b 1
)

:: Check if ML_inputs_1.txt exists
if not exist "%MACRO_DIR%\ML_inputs_1.txt" (
    echo Error: %MACRO_DIR%\ML_inputs_1.txt not found
    echo Cannot proceed without the template file
    pause
    exit /b 1
)

:: Count all ML_inputs files except ML_inputs_1.txt
set "count=0"
for %%f in ("%MACRO_DIR%\ML_inputs_*.txt") do (
    if /i not "%%~nxf"=="ML_inputs_1.txt" (
        set /a count+=1
    )
)

:: If no files to delete, exit
if %count%==0 (
    echo No additional ML_inputs files found in %MACRO_DIR%
    echo Only ML_inputs_1.txt exists, which will be kept
    pause
    exit /b 0
)

:: Show user what will be deleted
echo Found %count% additional ML_inputs files to delete:
echo.
for %%f in ("%MACRO_DIR%\ML_inputs_*.txt") do (
    if /i not "%%~nxf"=="ML_inputs_1.txt" (
        echo - %%~nxf
    )
)

rem :: Confirm before deleting
rem echo.
rem echo ML_inputs_1.txt will be kept as the template file.
rem set /p "confirm=Delete all other ML_inputs files? (y/n): "
rem if /i not "%confirm%"=="y" (
rem     echo Operation cancelled by user
rem     pause
rem     exit /b 0
rem )

rem :: Delete the files
rem echo.
rem echo Deleting files...
rem for %%f in ("%MACRO_DIR%\ML_inputs_*.txt") do (
rem     if /i not "%%~nxf"=="ML_inputs_1.txt" (
rem         echo Deleting: %%~nxf
rem         del "%%f"
rem     )
rem )

:: Automatic deletion without confirmation
echo.
echo ML_inputs_1.txt will be kept as the template file.
echo Automatically deleting all other ML_inputs files...

:: Delete the files
echo.
echo Deleting files...
for %%f in ("%MACRO_DIR%\ML_inputs_*.txt") do (
    if /i not "%%~nxf"=="ML_inputs_1.txt" (
        echo Deleting: %%~nxf
        del "%%f"
    )
)

echo.
echo Successfully deleted %count% ML_inputs files
echo Only ML_inputs_1.txt remains in the %MACRO_DIR% folder
pause
exit /b 0