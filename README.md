# 🐜 Ant Detection System

An intelligent application for detecting and counting ants in images using 
machine learning.

## 📥 Installation (Easy - 3 Steps!)

### For Mac Users:

1. **Install Python** (if you don't have it):
   - Download from: https://www.python.org/downloads/
   - Get version 3.8 or higher
   - During install, check "Add Python to PATH"

2. **Download required files**:
   - Download these files and save them in the same folder:
     - `ant_detection_app.py`
     - `install_and_run.py`
     - `requirements.txt`
     - `run_ant_detector.command` (Mac launcher)
  
3. **Run the application**:
   - Double-click `run_ant_detector.command`
   - The first time takes 2-3 minutes to install packages
   - After that, it launches instantly!

### For Windows Users:

1. **Install Python** (if you don't have it):
   - Download from: https://www.python.org/downloads/
   - Get version 3.8 or higher
   - **IMPORTANT:** Check "Add Python to PATH" during installation

2. **Download required files**:
   - Download these files and save them in the same folder:
     - `ant_detection_app.py`
     - `install_and_run.py`
     - `requirements.txt`
     - `run_ant_detector.bat` (Windows launcher)
  
3. **Run the application**:
   - Double-click `run_ant_detector.bat`
   - The first time takes 2-3 minutes to install packages
   - After that, it launches instantly!

## 💻 System Requirements

- **Operating System:** macOS 10.13+ or Windows 10+
- **Python:** 3.8 or higher (will be installed in step 1)
- **RAM:** 4 GB minimum, 8 GB recommended
- **Storage:** 500 MB free space

## 🚀 Quick Start

1. **Annotate Images:** Load images and mark ant locations
2. **Train Model:** Train the detection model on your annotations
3. **Test & Evaluate:** Assess model performance
4. **Batch Predict:** Process multiple images automatically

## 📋 Features

- 🎯 Interactive image annotation tool
- 🤖 Random Forest machine learning
- 📊 Comprehensive evaluation metrics
- 🔄 Batch processing capabilities
- 💾 Multiple export formats (CSV, JSON)

## ❓ Troubleshooting

### Mac Users

**App doesn't open when double-clicking?**

**Option 1: Fix the permissions**
1. Right Click `run_ant_detector.command`
2. Select "Open"
3. Confirm you want to open when the warning label pops up

**Option 2: Fix the permissions**
1. Open Terminal (press `Cmd + Space`, type "terminal", press Enter)
2. Type: `cd ` (with a space after cd)
3. Drag the folder containing your files into Terminal
4. Press Enter
5. Type: `chmod +x run_ant_detector.command` and press Enter
6. Now double-click `run_ant_detector.command` again

**Option 3: Run from Terminal directly**
1. Open Terminal
2. Type: `cd ` (with a space after cd)
3. Drag the folder containing your files into Terminal
4. Press Enter
5. Type: `python3 install_and_run.py` and press Enter

### Windows Users

**App doesn't open when double-clicking?**
1. Right-click `run_ant_detector.bat` and select "Run as administrator"
2. If that doesn't work, open Command Prompt:
   - Press `Win + R`, type `cmd`, press Enter
   - Type: `cd ` (with a space after cd)
   - Drag the folder containing your files into Command Prompt
   - Press Enter
   - Type: `python install_and_run.py` and press Enter

### Both Platforms

**"Python not found" error?**
- Make sure you installed Python and checked "Add to PATH"
- Restart your computer after installing Python
- Try `python3` instead of `python` (Mac) or vice versa (Windows)

**Packages fail to install?**
- Check your internet connection
- Try running: `pip install --upgrade pip` first

## 📧 Support

For issues or questions, contact davidlandis2150@gmail.com
