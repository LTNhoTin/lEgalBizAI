#!/usr/bin/env python3
"""
Automated training launcher for GPT-OSS 20B Fine-tuning
This script handles all setup and runs the complete training pipeline
"""

import os
import sys
import subprocess
from datetime import datetime

def activate_conda_env(env_name):
    """Activate conda environment"""
    try:
        # Check if conda is available
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️ Conda not found. Continuing with current environment...")
            return False
        
        print(f"🐍 Activating conda environment: {env_name}")
        
        # For Python scripts, we need to use the conda environment's Python
        conda_env_path = subprocess.run(
            ['conda', 'info', '--envs'], 
            capture_output=True, text=True
        ).stdout
        
        # Find the environment path
        for line in conda_env_path.split('\n'):
            if env_name in line and not line.startswith('#'):
                env_path = line.split()[-1]
                python_path = os.path.join(env_path, 'bin', 'python')
                if os.path.exists(python_path):
                    print(f"✅ Found conda environment at: {env_path}")
                    return python_path
        
        print(f"⚠️ Conda environment '{env_name}' not found. Using current environment...")
        return False
        
    except Exception as e:
        print(f"⚠️ Error activating conda environment: {e}")
        return False

def setup_wandb():
    """Setup Wandb if needed"""
    try:
        print("📊 Setting up Weights & Biases...")
        subprocess.run([sys.executable, "setup_wandb.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Wandb setup failed. Training will continue without logging.")
        return False
    except FileNotFoundError:
        print("⚠️ setup_wandb.py not found. Skipping wandb setup.")
        return False

def main():
    """Main launcher function"""
    print("🚀 GPT-OSS 20B Automated Training Launcher")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Setup Wandb
    setup_wandb()
    
    # Try to activate conda environment
    conda_python = activate_conda_env("gptoss-finetune")
    python_cmd = conda_python if conda_python else sys.executable
    
    # Run the main training script
    print("\n🏋️ Starting main training pipeline...")
    try:
        cmd = [python_cmd, "main.py", "--clear-cache"]
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Training completed successfully!")
        print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()