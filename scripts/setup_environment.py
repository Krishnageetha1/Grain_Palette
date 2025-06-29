"""
Environment Setup Script for Rice Classification Project
Run this script to set up your environment and install dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'models', 'static', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    print("🚀 Setting up Rice Classification Environment")
    print("=" * 50)
    
    # Required packages
    packages = [
        "Flask==2.3.3",
        "tensorflow==2.13.0",
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "Pillow==10.0.1",
        "matplotlib==3.7.2",
        "Werkzeug==2.3.7"
    ]
    
    print("📦 Installing required packages...")
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    print("\n✅ Environment setup completed!")
    print("\nNext steps:")
    print("1. Run 'python train_model.py' to train the model (optional)")
    print("2. Run 'python app.py' to start the web application")
    print("3. Open your browser and go to http://localhost:5000")

if __name__ == "__main__":
    main()
