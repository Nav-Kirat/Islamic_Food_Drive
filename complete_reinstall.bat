@echo off
echo ========================================================
echo COMPLETE REINSTALLATION OF NUMPY AND PANDAS
echo ========================================================
echo.
echo This script will:
echo 1. Completely remove NumPy, Pandas and related packages
echo 2. Install compatible versions in the correct order
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause > nul

:: Uninstall packages
echo.
echo Uninstalling packages...
pip uninstall -y numpy pandas streamlit scikit-learn matplotlib seaborn

:: Clean pip cache
echo.
echo Cleaning pip cache...
pip cache purge

:: Install packages in the correct order
echo.
echo Installing NumPy...
pip install numpy==1.21.6

echo Installing Pandas...
pip install pandas==1.3.5

echo Installing other dependencies...
pip install streamlit==1.22.0 scikit-learn==1.0.2 matplotlib==3.5.3 seaborn==0.12.2 openpyxl==3.1.2

echo.
echo ========================================================
echo REINSTALLATION COMPLETE!
echo ========================================================
echo.
echo Try running your app with:
echo streamlit run overview.py
echo.
echo If you still encounter issues, please try the setup_venv.bat
echo or setup_conda_env.bat scripts to create an isolated environment.
echo.
echo Press any key to exit...
pause > nul