@echo off
echo ========================================================
echo ALIGNING ENVIRONMENT WITH MODEL REQUIREMENTS
echo ========================================================
echo.
echo This script will:
echo 1. Uninstall current NumPy and Pandas
echo 2. Install NumPy 2.0.2 and compatible Pandas version
echo 3. Reinstall other dependencies
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause > nul

:: Uninstall packages
echo.
echo Uninstalling current NumPy and Pandas...
pip uninstall -y numpy pandas

:: Clean pip cache
echo.
echo Cleaning pip cache...
pip cache purge

:: Install NumPy 2.0.2 first
echo.
echo Installing NumPy 2.0.2...
pip install numpy==2.0.2

:: Install compatible Pandas version
echo.
echo Installing compatible Pandas version...
pip install pandas==2.1.4

:: Reinstall streamlit and other dependencies
echo.
echo Reinstalling other dependencies...
pip install streamlit==1.32.0 scikit-learn==1.4.0 matplotlib==3.8.2 seaborn==0.13.1 openpyxl==3.1.2 sentence-transformers==2.2.2 transformers==4.38.1 torch==2.2.0

echo.
echo ========================================================
echo INSTALLATION COMPLETE!
echo ========================================================
echo.
echo Try running your Streamlit app with:
echo streamlit run overview.py
echo.
echo Press any key to exit...
pause > nul