@echo off

echo ==========================================
echo PINN Modeling Environment Setup
echo ==========================================
echo.

set ENV_NAME=pinn-modeling

REM Initialize Conda for this batch file
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Could not find Miniconda.
    echo.
    echo Please check where Conda is installed.
    echo.
    pause
    exit /b 1
)

echo Conda found.
echo.

echo Updating environment: %ENV_NAME%
echo.

echo Removing old environment...
call conda env remove -n %ENV_NAME%

echo Creating updated environment...
call conda env create -f environment.yml

call conda activate pinn-modeling

python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126

@REM  call conda env update -n %ENV_NAME% -f environment.yml --prune

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to create/update the Conda environment.
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================
echo.

echo Testing environment...
echo.

call conda run -n %ENV_NAME% python test_environment.py

echo.
echo ==========================================
echo Setup and testing complete.
echo ==========================================
echo.

pause