@echo off
echo ========================================================
echo CREATING FRESH VIRTUAL ENVIRONMENT FOR STREAMLIT APP
echo ========================================================
echo.
echo This script will:
echo 1. Create a new virtual environment 'venv'
echo 2. Install compatible packages
echo 3. Provide instructions to activate and run your app
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause > nul

:: Create a new virtual environment
echo.
echo Creating new virtual environment 'venv'...
python -m venv venv

:: Activate the environment and install packages
echo.
echo Activating environment and installing packages...
call venv\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install packages in the correct order
echo Installing NumPy...
python -m pip install numpy==1.23.5

echo Installing Pandas...
python -m pip install pandas==1.5.3

echo Installing other dependencies...
python -m pip install streamlit==1.22.0 scikit-learn==1.2.2 matplotlib==3.7.1 seaborn==0.12.2 openpyxl==3.1.2 sentence-transformers==2.2.2 transformers==4.28.1 torch==2.0.0

echo.
echo ========================================================
echo SETUP COMPLETE!
echo ========================================================
echo.
echo To run your app:
echo 1. Open a new command prompt
echo 2. Navigate to your project folder: cd %CD%
echo 3. Activate the environment: venv\Scripts\activate
echo 4. Run: streamlit run overview.py
echo.
echo Press any key to exit...
pause > nul