@echo off
echo ========================================
echo FIXING NUMPY/PANDAS COMPATIBILITY ISSUE
echo ========================================
echo.
echo This will fix the "numpy.dtype size changed" error
echo.
python fix_numpy_pandas.py
echo.
echo If the fix is complete, try running:
echo streamlit run pages/Machine_learning.py
echo.
pause