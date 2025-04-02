"""
Complete fix for NumPy/Pandas compatibility issues.
This script will:
1. Uninstall NumPy, Pandas, and related packages
2. Install compatible versions in the correct order
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and print its output"""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    return result.returncode == 0

def main():
    print("=" * 80)
    print("FIXING NUMPY/PANDAS COMPATIBILITY ISSUE")
    print("=" * 80)
    
    # Step 1: Uninstall problematic packages
    print("\nStep 1: Uninstalling problematic packages...")
    packages_to_uninstall = [
        "numpy", 
        "pandas", 
        "streamlit", 
        "scikit-learn", 
        "matplotlib",
        "seaborn"
    ]
    
    for package in packages_to_uninstall:
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", package])
    
    # Step 2: Install NumPy first
    print("\nStep 2: Installing NumPy...")
    run_command([sys.executable, "-m", "pip", "install", "numpy==1.23.5"])
    
    # Step 3: Install Pandas
    print("\nStep 3: Installing Pandas...")
    run_command([sys.executable, "-m", "pip", "install", "pandas==1.5.3"])
    
    # Step 4: Install other dependencies
    print("\nStep 4: Installing other dependencies...")
    dependencies = [
        "streamlit==1.22.0",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "openpyxl==3.1.2",
        "sentence-transformers==2.2.2",
        "transformers==4.28.1",
        "torch==2.0.0"
    ]
    
    for dependency in dependencies:
        run_command([sys.executable, "-m", "pip", "install", dependency])
    
    print("\n" + "=" * 80)
    print("FIX COMPLETED!")
    print("=" * 80)
    print("\nTry running your Streamlit app again with:")
    print("streamlit run pages/Machine_learning.py")
    print("\nIf you still encounter issues, try creating a new virtual environment:")
    print("python -m venv new_env")
    print("new_env\\Scripts\\activate")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()