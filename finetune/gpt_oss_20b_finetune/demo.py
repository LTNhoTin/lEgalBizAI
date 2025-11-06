#!/usr/bin/env python3
"""
Demo script - Quick check các components
"""
import os
import sys
import json

def demo_config():
    """Demo config loading"""
    print("="*80)
    print("1. TESTING CONFIG")
    print("="*80)
    
    try:
        from config.config import config
        
        print("✅ Config loaded successfully!")
        print(f"   Model: {config.model.model_name}")
        print(f"   LoRA rank: {config.lora.r}")
        print(f"   Max samples: {config.data.max_samples}")
        print(f"   Epochs: {config.training.num_train_epochs}")
        print(f"   Wandb project: {config.wandb.project}")
        
        # Test mode config
        test_config = config.get_test_config()
        print(f"\n   Test mode LoRA rank: {test_config.lora.r}")
        print(f"   Test mode samples: {test_config.data.max_samples}")
        
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def demo_data_loader():
    """Demo data loader"""
    print("\n" + "="*80)
    print("2. TESTING DATA LOADER")
    print("="*80)
    
    try:
        from src.data_loader import LegalDataLoader
        from config.config import config
        
        # Check if data exists
        if not os.path.exists(config.data.data_path):
            print(f"⚠️  Data not found: {config.data.data_path}")
            print("   (This is OK for demo, but needed for training)")
            return True
        
        print(f"✅ Data file found: {config.data.data_path}")
        
        # Load a few samples
        with open(config.data.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Total samples in data: {len(data):,}")
        print(f"   Will sample: {config.data.max_samples}")
        
        # Show sample
        if data:
            sample = data[0]
            print(f"\n   Sample question: {sample['question'][:100]}...")
            print(f"   Sample answer length: {len(sample['answer'])} chars")
        
        print("\n✅ Data loader check passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False


def demo_model_imports():
    """Demo model imports"""
    print("\n" + "="*80)
    print("3. TESTING MODEL IMPORTS")
    print("="*80)
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM: {vram:.2f} GB")
        else:
            print("⚠️  CUDA not available (OK for demo)")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ Transformers imported")
        
        from peft import LoraConfig, get_peft_model
        print("✅ PEFT imported")
        
        import bitsandbytes
        print(f"✅ BitsAndBytes {bitsandbytes.__version__}")
        
        from datasets import Dataset
        print("✅ Datasets imported")
        
        import wandb
        print(f"✅ Wandb {wandb.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n   Run: pip install -r requirements.txt")
        return False


def demo_trainer():
    """Demo trainer"""
    print("\n" + "="*80)
    print("4. TESTING TRAINER")
    print("="*80)
    
    try:
        from src.trainer import GPTOSSTrainer
        from config.config import config
        
        # Create trainer instance (không load model)
        trainer = GPTOSSTrainer(config)
        
        print("✅ Trainer class loaded successfully!")
        print("   (Model not loaded in demo mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer error: {e}")
        return False


def demo_scripts():
    """Check scripts exist"""
    print("\n" + "="*80)
    print("5. CHECKING SCRIPTS")
    print("="*80)
    
    scripts = [
        ('main.py', 'Main training script'),
        ('inference.py', 'Inference script'),
        ('check_env.py', 'Environment check'),
        ('setup.sh', 'Setup script'),
    ]
    
    all_ok = True
    for script, desc in scripts:
        if os.path.exists(script):
            print(f"✅ {script} - {desc}")
        else:
            print(f"❌ {script} - NOT FOUND")
            all_ok = False
    
    return all_ok


def demo_docs():
    """Check documentation"""
    print("\n" + "="*80)
    print("6. CHECKING DOCUMENTATION")
    print("="*80)
    
    docs = [
        'README.md',
        'QUICK_START.md',
        'PROJECT_OVERVIEW.md',
        'SETUP_SUMMARY.md',
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc) / 1024
            print(f"✅ {doc} ({size:.1f} KB)")
        else:
            print(f"⚠️  {doc} - Not found")
    
    return True


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("GPT-OSS 20B FINE-TUNING - DEMO CHECK")
    print("="*80)
    print("\nKiểm tra nhanh tất cả components...\n")
    
    results = []
    
    results.append(("Config", demo_config()))
    results.append(("Data Loader", demo_data_loader()))
    results.append(("Model Imports", demo_model_imports()))
    results.append(("Trainer", demo_trainer()))
    results.append(("Scripts", demo_scripts()))
    results.append(("Documentation", demo_docs()))
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r for _, r in results)
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check environment: python check_env.py")
        print("  3. Test training: python main.py --test")
        print("  4. Full training: python main.py")
        print("\n📚 See QUICK_START.md for more info")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above.")
        print("Most likely you need to:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

