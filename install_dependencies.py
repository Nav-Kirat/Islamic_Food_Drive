"""
Comprehensive Dependency Installation Script for Islamic Food Drive Project

This script helps you install and manage project dependencies with multiple options.
"""

import subprocess
import sys
import platform
import os

def run_command(command):
    """Run a shell command and print its output"""
    print(f"\n>>> Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.stderr)

def check_python_version():
    """Check if the Python version is compatible"""
    print(f"Current Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")

def install_dependencies():
    """Install project dependencies"""
    # Uninstall existing packages to prevent conflicts
    packages_to_uninstall = [
        "numpy", "pandas", "streamlit", "scikit-learn", 
        "matplotlib", "seaborn", "openpyxl"
    ]
    
    for package in packages_to_uninstall:
        try:
            run_command([sys.executable, "-m", "pip", "uninstall", "-y", package])
        except:
            pass
    
    # Clean pip cache
    run_command([sys.executable, "-m", "pip", "cache", "purge"])
    
    # Install dependencies in specific order
    dependencies = [
        "numpy==2.0.2",
        "pandas==2.1.4",
        "streamlit==1.32.0",
        "scikit-learn==1.4.0",
        "matplotlib==3.8.2",
        "seaborn==0.13.1",
        "openpyxl==3.1.2",
        "sentence-transformers==2.2.2",
        "transformers==4.38.1",
        "torch==2.2.0"
    ]
    
    for dependency in dependencies:
        run_command([sys.executable, "-m", "pip", "install", dependency])

def verify_installation():
    """Verify the installed packages"""
    try:
        import numpy
        import pandas
        import streamlit
        
        print("\n✅ Verification Successful!")
        print(f"NumPy Version: {numpy.__version__}")
        print(f"Pandas Version: {pandas.__version__}")
        print(f"Streamlit Version: {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Verification Failed: {e}")

def main():
    print("=" * 50)
    print("🍽️ Islamic Food Drive Dependency Installer")
    print("=" * 50)
    
    check_python_version()
    
    print("\n1. Installing Dependencies...")
    install_dependencies()
    
    print("\n2. Verifying Installation...")
    verify_installation()
    
    print("\n🚀 Installation Complete!")
    print("Run your Streamlit app with: streamlit run overview.py")

if __name__ == "__main__":
    main()