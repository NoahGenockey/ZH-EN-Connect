# LinguaBridge Local - Build Desktop Application
# This script creates a standalone executable

import PyInstaller.__main__
import os
import sys

def build_executable():
    """Build standalone desktop application."""
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # PyInstaller options
    options = [
        'src/app_gui.py',  # Main script
        '--name=LinguaBridge',  # Application name
        '--onefile',  # Single executable
        '--windowed',  # No console window
        '--icon=assets/icon.ico',  # App icon (if exists)
        
        # Include data files
        '--add-data=config.yaml;.',
        '--add-data=models;models',
        '--add-data=src;src',
        
        # Hidden imports
        '--hidden-import=tkinter',
        '--hidden-import=transformers',
        '--hidden-import=torch',
        '--hidden-import=jieba',
        
        # Output directory
        '--distpath=build/dist',
        '--workpath=build/work',
        '--specpath=build',
        
        # Clean build
        '--clean',
        
        # Exclude unnecessary modules
        '--exclude-module=matplotlib',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
    ]
    
    print("=" * 60)
    print(" Building LinguaBridge Desktop Application")
    print("=" * 60)
    print()
    print("This will create a standalone executable...")
    print(f"Output: build/dist/LinguaBridge.exe")
    print()
    
    try:
        PyInstaller.__main__.run(options)
        print()
        print("=" * 60)
        print(" ✅ Build Complete!")
        print("=" * 60)
        print()
        print(f"Executable location: {os.path.join(project_root, 'build', 'dist', 'LinguaBridge.exe')}")
        print()
        print("You can now distribute this single .exe file!")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
