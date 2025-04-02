@echo off
echo ========================================================
echo CREATING ISOLATED CONDA ENVIRONMENT FOR STREAMLIT APP
echo ========================================================
echo.
echo This script will:
echo 1. Download and install Miniconda (if not already installed)
echo 2. Create a new conda environment with compatible packages
echo 3. Provide instructions to activate and run your app
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause > nul

:: Check if conda is already installed
where conda > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda not found. Downloading Miniconda installer...
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    echo Installing Miniconda...
    start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
    set PATH=%UserProfile%\Miniconda3;%UserProfile%\Miniconda3\Scripts;%PATH%
) else (
    echo Conda is already installed.
)

:: Create a new conda environment
echo.
echo Creating new conda environment 'islamic_food_app'...
call conda create -y -n islamic_food_app python=3.9

:: Activate the environment and install packages
echo.
echo Installing required packages...
call conda activate islamic_food_app
call conda install -y numpy=1.23.5 pandas=1.5.3 matplotlib=3.7.1 scikit-learn=1.2.2 seaborn=0.12.2
call pip install streamlit==1.22.0 openpyxl==3.1.2 sentence-transformers==2.2.2 transformers==4.28.1 torch==2.0.0

echo.
echo ========================================================
echo SETUP COMPLETE!
echo ========================================================
echo.
echo To run your app:
echo 1. Open a new command prompt
echo 2. Run: conda activate islamic_food_app
echo 3. Navigate to your project folder: cd %CD%
echo 4. Run: streamlit run overview.py
echo.
echo Press any key to exit...
pause > nul