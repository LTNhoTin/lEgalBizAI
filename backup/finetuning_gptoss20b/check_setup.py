#!/usr/bin/env python3
"""
Dependency and setup checker for GPT-OSS 20B Fine-tuning
"""

import sys
import subprocess
import importlib
import torch
import os
from datetime import datetime

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_cuda():
    """Check CUDA availability"""
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"🔧 CUDA version: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return True
    else:
        print("❌ CUDA not available")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: not installed")
        return False

def check_conda():
    """Check conda environment"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Conda: {result.stdout.strip()}")
            
            # Check for gptoss-finetune environment
            env_result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
            if 'gptoss-finetune' in env_result.stdout:
                print("✅ Conda environment 'gptoss-finetune' found")
            else:
                print("⚠️ Conda environment 'gptoss-finetune' not found")
            return True
        else:
            print("❌ Conda not available")
            return False
    except FileNotFoundError:
        print("❌ Conda not found")
        return False

def check_wandb():
    """Check Wandb setup"""
    try:
        import wandb
        print(f"✅ Wandb: {wandb.__version__}")
        
        # Check if logged in
        try:
            if wandb.api.api_key:
                print("✅ Wandb: logged in")
            else:
                print("⚠️ Wandb: not logged in (run 'wandb login')")
        except:
            print("⚠️ Wandb: login status unknown")
        
        return True
    except ImportError:
        print("❌ Wandb: not installed")
        return False

def check_files():
    """Check required files"""
    required_files = [
        'main.py',
        'config/config.py',
        'src/trainer.py',
        'src/data_loader.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: exists")
        else:
            print(f"❌ {file_path}: missing")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check dataset availability"""
    from config.config import config
    
    dataset_path = config.data.dataset_path
    if os.path.exists(dataset_path):
        print(f"✅ Dataset: {dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        return False

def main():
    """Main check function"""
    print("🔍 GPT-OSS 20B Fine-tuning Setup Checker")
    print("=" * 60)
    print(f"🕐 Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = []
    
    # Basic checks
    print("📋 Basic Requirements:")
    checks.append(check_python_version())
    checks.append(check_cuda())
    print()
    
    # Package checks
    print("📦 Package Dependencies:")
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('unsloth', 'unsloth'),
        ('trl', 'trl'),
        ('datasets', 'datasets'),
        ('peft', 'peft'),
    ]
    
    for pkg_name, import_name in packages:
        checks.append(check_package(pkg_name, import_name))
    
    checks.append(check_wandb())
    print()
    
    # Environment checks
    print("🐍 Environment:")
    check_conda()
    print()
    
    # File checks
    print("📁 Files:")
    checks.append(check_files())
    print()
    
    # Dataset check
    print("📊 Dataset:")
    try:
        checks.append(check_dataset())
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        checks.append(False)
    print()
    
    # Summary
    print("📋 Summary:")
    passed = sum(checks)
    total = len(checks)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! Ready for training.")
        return True
    else:
        print("⚠️ Some checks failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)