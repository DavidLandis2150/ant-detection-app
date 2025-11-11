import subprocess
import sys
import os

def install_and_run():
    """Install dependencies and run the application"""
    
    print("🐜 Ant Detection System - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Your version: Python {sys.version}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\n📦 Installing required packages...")
    print("This may take a few minutes on first run...\n")
    
    # Updated package list for the new Tkinter application
    packages = [
        "numpy>=1.26.0",
        "opencv-python>=4.9.0.80", 
        "Pillow>=10.3.0",
        "scikit-learn>=1.4.0",
        "scikit-image>=0.22.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "psutil>=5.9.6"
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                  capture_output=True, text=True, check=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}")
            print(f"     Error: {e.stderr.strip()}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️ Warning: Failed to install {len(failed_packages)} packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\nThe application may not work correctly.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Setup cancelled.")
            input("Press Enter to exit...")
            sys.exit(1)
    
    print("\n✅ All dependencies installed successfully!")
    
    # Check if the main application file exists
    app_file = "ant_detection_app.py"
    if not os.path.exists(app_file):
        print(f"\n❌ Error: {app_file} not found in current directory")
        print("Please ensure the application file is in the same folder as this installer.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Run the application
    print("\n🚀 Launching Ant Detection System...")
    print("=" * 40)
    print("Note: The application window should open shortly...")
    print("If no window appears, check the console for error messages.\n")
    
    try:
        # Use check=True to catch application errors
        subprocess.run([sys.executable, app_file], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Application exited with error code {e.returncode}")
        print("This may indicate a problem with dependencies or the application code.")
        input("Press Enter to exit...")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find Python executable or {app_file}")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ Unexpected error running application: {e}")
        input("Press Enter to exit...")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    import_tests = [
        ("numpy", "import numpy as np"),
        ("OpenCV", "import cv2"),
        ("PIL", "from PIL import Image"),
        ("scikit-learn", "from sklearn.ensemble import RandomForestClassifier"),
        ("scikit-image", "from skimage.feature import hog"),
        ("pandas", "import pandas as pd"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("psutil", "import psutil"),
        ("tkinter", "import tkinter as tk")
    ]
    
    failed_imports = []
    
    for name, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️ Warning: {len(failed_imports)} packages failed to import")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

if __name__ == "__main__":
    try:
        install_and_run()
        
        # Test imports after installation
        if test_imports():
            print("\n🎉 Setup completed successfully!")
            print("You can now run 'python ant_detection_app.py' directly in the future.")
        else:
            print("\n⚠️ Setup completed with warnings. Some features may not work correctly.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        input("Press Enter to exit...")