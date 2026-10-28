@echo off
setlocal

echo ==========================================
echo PINN Modeling Environment Setup
echo ==========================================
echo.

set ENV_NAME=pinn-modeling
set TORCH_INDEX=https://download.pytorch.org/whl/cu126

REM Run from this file's folder so environment.yml and test_environment.py are found
cd /d "%~dp0"

REM ------------------------------------------------------------------
REM 1. Find Conda: common install folders first, then conda.exe on PATH
REM ------------------------------------------------------------------
set "CONDA_ACTIVATE="
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined CONDA_ACTIVATE if exist "%%~D\Scripts\activate.bat" set "CONDA_ACTIVATE=%%~D\Scripts\activate.bat"
)
if not defined CONDA_ACTIVATE (
    for /f "delims=" %%C in ('where conda.exe 2^>nul') do (
        if not defined CONDA_ACTIVATE if exist "%%~dpCactivate.bat" set "CONDA_ACTIVATE=%%~dpCactivate.bat"
    )
)
if not defined CONDA_ACTIVATE (
    echo ERROR: Could not find Conda. Install Miniconda, or run this file from an Anaconda Prompt.
    goto :fail
)

call "%CONDA_ACTIVATE%"
if errorlevel 1 (
    echo ERROR: Could not initialize Conda from %CONDA_ACTIVATE%
    goto :fail
)
echo Conda found: %CONDA_ACTIVATE%
echo.

REM ------------------------------------------------------------------
REM 2. Remove an existing environment, only after asking
REM ------------------------------------------------------------------
call conda env list | findstr /r /c:"^%ENV_NAME% " >nul
if not errorlevel 1 (
    echo The environment "%ENV_NAME%" already exists.
    choice /m "Delete it and create it again from environment.yml"
    if errorlevel 2 (
        echo Setup cancelled. The existing environment was not changed.
        goto :done
    )
    echo Removing old environment...
    call conda env remove -n %ENV_NAME% -y
    if errorlevel 1 (
        echo ERROR: Could not remove the existing environment.
        goto :fail
    )
)

REM ------------------------------------------------------------------
REM 3. Create the environment from environment.yml
REM ------------------------------------------------------------------
echo Creating environment "%ENV_NAME%" from environment.yml...
call conda env create -f environment.yml -n %ENV_NAME%
if errorlevel 1 (
    echo ERROR: conda env create failed.
    goto :fail
)

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo ERROR: Could not activate "%ENV_NAME%".
    goto :fail
)

REM ------------------------------------------------------------------
REM 4. PyTorch with CUDA 12.6 (from the PyTorch index, not conda-forge)
REM ------------------------------------------------------------------
echo.
echo Installing PyTorch (CUDA 12.6 build)...
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo ERROR: PyTorch installation failed.
    goto :fail
)

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================
echo.

REM ------------------------------------------------------------------
REM 5. Test the environment
REM ------------------------------------------------------------------
echo Testing environment...
echo.
python test_environment.py

echo.
echo ==========================================
echo Setup and testing complete.
echo ==========================================

:done
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo Setup did not finish. See the message above.
pause
endlocal
exit /b 1
