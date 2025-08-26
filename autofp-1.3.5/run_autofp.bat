@echo off
echo Starting AutoFP refinement with Python 2.7
call conda activate py27
python autofp_fs_unselect_GUI_suppressed.py
IF %ERRORLEVEL% NEQ 0 echo AutoFP returned error code: %ERRORLEVEL%
call conda deactivate
echo AutoFP process completed
