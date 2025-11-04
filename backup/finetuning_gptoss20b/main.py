#!/usr/bin/env python3
"""Main script to run GPT-OSS 20B finetuning"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import GPTTrainer
from config.config import config

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(config.data.dataset_path):
        print(f"Dataset not found at {config.data.dataset_path}")
        print("Please ensure the dataset file exists before running training.")
        sys.exit(1)
    
    print(f"Dataset found: {config.data.dataset_path}")
    print(f"Models directory: {config.models_dir}")
    print(f"Logs directory: {config.logs_dir}")
    print(f"Output directory: {config.training.output_dir}")

def print_config():
    """Print current configuration"""
    print("\nCurrent Configuration:")
    print("-" * 40)
    print(f"Model: {config.model.model_name}")
    print(f"Max sequence length: {config.model.max_seq_length}")
    print(f"LoRA rank: {config.model.lora_r}")
    print(f"LoRA alpha: {config.model.lora_alpha}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Dataset: {config.data.dataset_path}")
    print("-" * 40)

def main():
    """Main function to run GPT-OSS 20B fine-tuning"""
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS 20B model")
    parser.add_argument(
        "--skip-setup", 
        action="store_true", 
        help="Skip environment setup checks"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="finetuned_model",
        help="Name for the saved model (default: finetuned_model)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--clear-cache", 
        choices=["all", "model", "dataset"], 
        help="Clear cache before training"
    )
    parser.add_argument(
        "--cache-info", 
        action="store_true", 
        help="Show cache information"
    )
    parser.add_argument(
        "--force-reload", 
        action="store_true", 
        help="Force reload model and dataset (ignore cache)"
    )
    parser.add_argument(
        "--skip-conda", 
        action="store_true",
        help="Skip conda environment activation"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    print("🚀 Starting GPT-OSS 20B Fine-tuning Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.skip_setup:
        setup_environment()
    
    print_config()
    
    # Confirm before starting
    if not args.skip_setup:
        response = input("\n❓ Do you want to start training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            sys.exit(0)
    
    try:
        # Initialize trainer
        trainer = GPTTrainer()
        
        # Handle cache operations
        if args.cache_info:
            trainer.get_cache_info()
            return
        
        # Activate conda environment if not skipped
        if not args.skip_conda:
            print("\n🐍 Activating conda environment 'gptoss-finetune'...")
            try:
                trainer.activate_conda_env("gptoss-finetune")
                print("✅ Conda environment activated successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not activate conda environment: {e}")
                print("Continuing with current environment...")
        
        # Clear VRAM and cache
        print("\n🧹 Clearing VRAM and cache...")
        trainer.clear_vram()
        if args.clear_cache:
            trainer.clear_cache(args.clear_cache)
        
        # Setup Wandb
        print("\n📊 Setting up Weights & Biases logging...")
        try:
            trainer.setup_wandb()
            print("✅ Wandb initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Wandb: {e}")
            print("Training will continue without Wandb logging...")
        
        # Run training pipeline with force_reload flag
        model_path = trainer.run_full_training_pipeline(force_reload=args.force_reload)
        
        # Save with custom name if provided
        if args.model_name != "finetuned_model":
            custom_path = trainer.save_model(args.model_name)
            print(f"\n📁 Model also saved with custom name: {custom_path}")
        
        # Final VRAM cleanup
        print("\n🧹 Final VRAM cleanup...")
        trainer.clear_vram()
        
        print("\n✅ Training pipeline completed successfully!")
        print(f"📁 Final model location: {model_path}")
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Instructions for next steps
        print("\n📝 Next steps:")
        print("1. Check Wandb dashboard for detailed metrics")
        print("2. Test your model with: python inference.py")
        print(f"3. Your model is saved at: {model_path}")
        print("4. Check training logs in the outputs directory")
        print("5. Deploy to production if satisfied")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check if you have enough GPU memory")
        print("2. Verify dataset format is correct")
        print("3. Ensure all dependencies are installed")
        print("4. Check the logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()