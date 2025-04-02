"""
Fix environment script for NumPy/Pandas compatibility issues.
Run this script to reinstall compatible versions of NumPy and Pandas.
"""

import subprocess
import sys
import os

def main():
    print("Starting environment fix for NumPy/Pandas compatibility...")
    
    # Uninstall current versions
    print("\n1. Uninstalling current NumPy and Pandas packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "pandas"])
    
    # Install specific versions
    print("\n2. Installing compatible versions...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "pandas==2.0.3"])
    
    # Install other requirements
    print("\n3. Installing other requirements from requirements.txt...")
    if os.path.exists("requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\n✅ Environment fix completed!")
    print("\nTry running your Streamlit app again with:")
    print("streamlit run pages/Machine_learning.py")

if __name__ == "__main__":
    main()