@echo off

REM Save the current directory
set CURRENT_DIR=%cd%

REM Change to the directory where the batch file is located
cd /d %~dp0

REM Activate the conda environment
call conda activate py27

REM Run the Python script to refresh the working directory
python scripts\refresh_datadirectory.py

REM Change back to the original directory
cd %CURRENT_DIR%

rem rem REM Add this line to prompt the user to press any key to continue
rem pause

REM Automatically close the command prompt window after execution
exit
