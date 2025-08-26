@echo off
REM **Npr=5**: **Pseudo-Voigt function**
REM * A simple combination of Gaussian and Lorentzian functions
REM * Uses parameters: U, V, W (for Gaussian width), X, Y (for Lorentzian width)
REM * Good for basic peak shape modeling
REM
REM **Npr=7**: **Thompson-Cox-Hastings (TCH) pseudo-Voigt**
REM * A more sophisticated profile function
REM * Better accounts for instrumental and sample contributions
REM * Uses additional parameters: GauSiz, LorSiz (for size broadening)
REM * More accurate for complex peak shapes and microstructural analysis
REM
REM **Key differences**:
REM * TCH (Npr=7) separates instrumental and sample broadening effects better
REM * TCH can model asymmetric peaks more accurately
REM * Pseudo-Voigt (Npr=5) is simpler but less flexible
REM * TCH is preferred for detailed microstructural analysis (crystallite size, strain)
REM


cd /d "%~dp0"
call conda activate
python -c "import re; f='autofp-1.3.5/diffpy/pyfullprof/contribution.py'; content=open(f,'r').read(); has_7='self.set(\"Npr\", 7)' in content; new_content=re.sub(r'self\.set\(\"Npr\",\s*7\)', 'self.set(\"Npr\", 5)', content) if has_7 else re.sub(r'self\.set\(\"Npr\",\s*5\)', 'self.set(\"Npr\", 7)', content); open(f,'w').write(new_content); print('Changed Npr from 7 (TCH) to 5 (Pseudo-Voigt)' if has_7 else 'Changed Npr from 5 (Pseudo-Voigt) to 7 (TCH)')"
pause