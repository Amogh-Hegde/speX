

# setup_system.py
import os
import subprocess
import sys

def setup_environment():
    """
    Sets up the environment for   assistant system, installing only
    the necessary packages and creating   directory structure.
    """
    print("Starting system setup...")

    # Create directory structure
    directories = [
        'datasets/faces',          # For face recognition dataset
        'models',                  # For pre-trained models
        'src/recognition',         # For recognition modules
        'src/utils',              # For utility functions
        'src/main'                # For main program files
    ]

    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created {directory}")

    # Install required packages
    print("\nInstalling required packages...")
    packages = [
        'numpy',
        'opencv-python',
        'mediapipe',
        'pytesseract',
        'pyttsx3',
        'speechrecognition',
        'cmake'  # Required for dlib installation
    ]

    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            continue

    # Special handling for dlib
    print("\nInstalling dlib (this might take a while)...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             'dlib', '--force-reinstall', '--no-cache-dir'])
    except subprocess.CalledProcessError as e:
        print("Error installing dlib. Please make sure you have Visual Studio Build Tools installed.")
        print("You can download them from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return

    print("\nSetup completed successfully!")
    print("\nDirectory structure created:")
    print("-----------------------------")
    for directory in directories:
        print(f"- {directory}")

    print("\nNext steps:")
    print("1. Download YOLO weights to the 'models' directory")
    print("2. Start collecting face data using face_dataset_builder.py")
    print("3. Run the main assistant using spex_assistant.py")

if __name__ == "__main__":
    setup_environment()