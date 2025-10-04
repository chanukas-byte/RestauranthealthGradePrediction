#!/usr/bin/env python3
"""
🏥 Restaurant Health Prediction App - Deployment Health Check
============================================================

This script validates that your app is ready for Streamlit Cloud deployment.
Run this before deploying to catch any potential issues.
"""

import os
import sys
from pathlib import Path
import importlib.util

def check_file_exists(file_path, description):
    """Check if a file exists and return status"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} can be imported")
        return True
    except ImportError:
        print(f"❌ {module_name} cannot be imported")
        return False

def check_file_size(file_path, max_size_mb=100):
    """Check if file size is within acceptable limits"""
    if not Path(file_path).exists():
        return False
    
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    if size_mb <= max_size_mb:
        print(f"✅ {file_path}: {size_mb:.1f}MB (within {max_size_mb}MB limit)")
        return True
    else:
        print(f"⚠️ {file_path}: {size_mb:.1f}MB (exceeds {max_size_mb}MB limit)")
        return False

def main():
    print("🏥 Restaurant Health Prediction App - Deployment Health Check")
    print("=" * 65)
    
    all_checks_passed = True
    
    # Check essential files
    print("\n📁 Checking Essential Files:")
    print("-" * 30)
    
    essential_files = [
        ("streamlit_app.py", "Main application file"),
        ("requirements.txt", "Python dependencies"),
        (".streamlit/config.toml", "Streamlit configuration"),
        ("data/cleaned_restaurant_dataset.csv", "Restaurant dataset"),
        ("models/restaurant_balanced_model.joblib", "Trained model"),
        ("README.md", "Project documentation"),
    ]
    
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # Check file sizes
    print("\n📊 Checking File Sizes:")
    print("-" * 25)
    
    large_files = [
        ("data/cleaned_restaurant_dataset.csv", 50),  # 50MB limit for data
        ("models/restaurant_balanced_model.joblib", 100),  # 100MB limit for model
    ]
    
    for file_path, max_size in large_files:
        if Path(file_path).exists():
            if not check_file_size(file_path, max_size):
                all_checks_passed = False
    
    # Check Python imports
    print("\n🐍 Checking Python Dependencies:")
    print("-" * 35)
    
    required_modules = [
        "streamlit",
        "pandas", 
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "joblib",
    ]
    
    for module in required_modules:
        if not check_import(module):
            all_checks_passed = False
    
    # Check git status
    print("\n📦 Checking Git Repository:")
    print("-" * 28)
    
    if Path(".git").exists():
        print("✅ Git repository initialized")
        
        # Check if there are uncommitted changes
        import subprocess
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            if result.stdout.strip():
                print("⚠️ Uncommitted changes detected:")
                print(result.stdout)
                print("💡 Run 'git add . && git commit -m \"Deploy prep\"' to commit changes")
            else:
                print("✅ Working directory clean")
        except subprocess.CalledProcessError:
            print("⚠️ Could not check git status")
    else:
        print("❌ Git repository not initialized")
        all_checks_passed = False
    
    # Check Streamlit app syntax
    print("\n🔍 Checking App Syntax:")
    print("-" * 25)
    
    try:
        import ast
        with open("streamlit_app.py", "r") as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ streamlit_app.py syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error in streamlit_app.py: {e}")
        all_checks_passed = False
    except Exception as e:
        print(f"⚠️ Could not validate syntax: {e}")
    
    # Summary
    print("\n" + "=" * 65)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! Your app is ready for deployment.")
        print("\n🚀 Next Steps:")
        print("1. Run: ./deploy.sh")
        print("2. Go to: https://share.streamlit.io")
        print("3. Deploy your app!")
        print("\n📚 See STREAMLIT_CLOUD_DEPLOYMENT.md for detailed instructions.")
    else:
        print("❌ Some checks failed. Please fix the issues above before deploying.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
