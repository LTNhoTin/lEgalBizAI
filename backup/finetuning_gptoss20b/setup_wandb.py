#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration
"""

import subprocess
import sys
import os

def install_wandb():
    """Install wandb if not available"""
    try:
        import wandb
        print("✅ Wandb already installed")
        return True
    except ImportError:
        print("📦 Installing Wandb...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
            print("✅ Wandb installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install wandb: {e}")
            return False

def setup_wandb_login():
    """Setup wandb login"""
    try:
        import wandb
        
        # Check if already logged in
        if wandb.api.api_key:
            print("✅ Already logged in to Wandb")
            return True
        
        print("🔑 Setting up Wandb login...")
        print("Please visit https://wandb.ai/authorize to get your API key")
        
        api_key = input("Enter your Wandb API key (or press Enter to skip): ").strip()
        
        if api_key:
            wandb.login(key=api_key)
            print("✅ Wandb login successful")
            return True
        else:
            print("⚠️ Skipping Wandb login. You can login later with: wandb login")
            return False
            
    except Exception as e:
        print(f"❌ Failed to setup wandb login: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Weights & Biases for GPT-OSS 20B Fine-tuning")
    print("=" * 60)
    
    # Install wandb
    if not install_wandb():
        print("❌ Failed to install wandb. Training will continue without logging.")
        return False
    
    # Setup login
    setup_wandb_login()
    
    print("\n✅ Wandb setup completed!")
    print("📊 Your training metrics will be logged to Wandb dashboard")
    print("🌐 Visit https://wandb.ai to view your experiments")
    
    return True

if __name__ == "__main__":
    main()