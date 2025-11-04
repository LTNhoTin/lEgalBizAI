#!/usr/bin/env python3
"""
Main script để fine-tune GPT-OSS 20B với LoRA/QLoRA
Sử dụng: 
    python main.py           # Training với full config
    python main.py --test    # Training với config nhỏ để test
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import wandb

# Add src to path
sys.path.append(os.path.dirname(__file__))

from config.config import config
from src.data_loader import LegalDataLoader
from src.trainer import GPTOSSTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(__file__), 
                'logs', 
                f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        )
    ]
)
logger = logging.getLogger(__name__)


def setup_wandb(cfg):
    """Setup Wandb logging"""
    logger.info("Setting up Wandb...")
    
    # Generate run name nếu chưa có
    if cfg.wandb.name is None:
        cfg.wandb.name = f"gpt-oss-legal-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Init wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config={
            "model": cfg.model.__dict__,
            "lora": cfg.lora.__dict__,
            "data": cfg.data.__dict__,
            "training": cfg.training.__dict__,
        }
    )
    
    logger.info(f"Wandb run: {cfg.wandb.name}")


def print_config(cfg):
    """In ra config hiện tại"""
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info("\n📦 MODEL CONFIG:")
    logger.info(f"  Model: {cfg.model.model_name}")
    logger.info(f"  4-bit: {cfg.model.load_in_4bit}")
    logger.info(f"  8-bit: {cfg.model.load_in_8bit}")
    
    logger.info("\n🎯 LORA CONFIG:")
    logger.info(f"  Rank (r): {cfg.lora.r}")
    logger.info(f"  Alpha: {cfg.lora.lora_alpha}")
    logger.info(f"  Target modules: {cfg.lora.target_modules}")
    logger.info(f"  Dropout: {cfg.lora.lora_dropout}")
    
    logger.info("\n📊 DATA CONFIG:")
    logger.info(f"  Data path: {cfg.data.data_path}")
    logger.info(f"  Max samples: {cfg.data.max_samples}")
    logger.info(f"  Train/Test split: {cfg.data.train_split}/{cfg.data.test_split}")
    logger.info(f"  Max length: {cfg.data.max_length}")
    
    logger.info("\n🏋️ TRAINING CONFIG:")
    logger.info(f"  Output dir: {cfg.training.output_dir}")
    logger.info(f"  Epochs: {cfg.training.num_train_epochs}")
    logger.info(f"  Batch size: {cfg.training.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {cfg.training.learning_rate}")
    logger.info(f"  Optimizer: {cfg.training.optim}")
    logger.info(f"  BF16: {cfg.training.bf16}")
    logger.info(f"  Gradient checkpointing: {cfg.training.gradient_checkpointing}")
    
    logger.info("\n📝 WANDB CONFIG:")
    logger.info(f"  Project: {cfg.wandb.project}")
    logger.info(f"  Run name: {cfg.wandb.name}")
    logger.info(f"  Tags: {cfg.wandb.tags}")
    
    logger.info("="*80 + "\n")


def check_environment():
    """Kiểm tra environment"""
    logger.info("Checking environment...")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
        logger.info(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("⚠️  CUDA not available! Training sẽ rất chậm.")
    
    # Check packages
    try:
        import transformers
        import peft
        import bitsandbytes
        import wandb
        logger.info(f"✅ transformers: {transformers.__version__}")
        logger.info(f"✅ peft: {peft.__version__}")
        logger.info(f"✅ bitsandbytes: {bitsandbytes.__version__}")
        logger.info(f"✅ wandb: {wandb.__version__}")
    except ImportError as e:
        logger.error(f"❌ Missing package: {e}")
        logger.error("Chạy: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("Environment check passed!\n")


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS 20B')
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Chạy training với config nhỏ để test'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Không sử dụng wandb logging'
    )
    args = parser.parse_args()
    
    # Get config
    if args.test:
        logger.info("🧪 TEST MODE - Sử dụng config nhỏ để test")
        cfg = config.get_test_config()
    else:
        logger.info("🚀 FULL TRAINING MODE")
        cfg = config
    
    # Print config
    print_config(cfg)
    
    # Check environment
    check_environment()
    
    # Setup wandb
    if not args.no_wandb:
        setup_wandb(cfg)
    else:
        cfg.training.report_to = "none"
        logger.info("Wandb logging disabled")
    
    try:
        # Step 1: Load data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        
        data_loader = LegalDataLoader(
            data_path=cfg.data.data_path,
            max_samples=cfg.data.max_samples,
            train_split=cfg.data.train_split,
            random_seed=cfg.data.random_seed
        )
        
        # Step 2: Setup model và tokenizer
        logger.info("\n" + "="*80)
        logger.info("STEP 2: SETTING UP MODEL AND TOKENIZER")
        logger.info("="*80)
        
        trainer_obj = GPTOSSTrainer(cfg)
        model, tokenizer = trainer_obj.setup_model_and_tokenizer()
        
        # Print sample
        logger.info("\n" + "="*80)
        logger.info("SAMPLE DATA CHECK")
        logger.info("="*80)
        data_loader.print_sample(tokenizer, cfg.data.prompt_template)
        
        # Step 3: Create datasets
        logger.info("\n" + "="*80)
        logger.info("STEP 3: CREATING DATASETS")
        logger.info("="*80)
        
        train_dataset, eval_dataset = data_loader.create_datasets(
            tokenizer=tokenizer,
            prompt_template=cfg.data.prompt_template,
            max_length=cfg.data.max_length
        )
        
        # Step 4: Create trainer
        logger.info("\n" + "="*80)
        logger.info("STEP 4: CREATING TRAINER")
        logger.info("="*80)
        
        trainer = trainer_obj.create_trainer(train_dataset, eval_dataset)
        
        # Step 5: Train
        logger.info("\n" + "="*80)
        logger.info("STEP 5: TRAINING")
        logger.info("="*80)
        
        trainer_obj.train()
        
        # Step 6: Evaluate
        logger.info("\n" + "="*80)
        logger.info("STEP 6: FINAL EVALUATION")
        logger.info("="*80)
        
        metrics = trainer_obj.evaluate()
        
        # Step 7: Save model
        logger.info("\n" + "="*80)
        logger.info("STEP 7: SAVING MODEL")
        logger.info("="*80)
        
        output_dir = trainer_obj.save_model()
        
        # Done!
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Model saved to: {output_dir}")
        logger.info("\nFinal metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*80 + "\n")
        
        if not args.no_wandb:
            wandb.finish()
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("❌ TRAINING FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*80 + "\n")
        
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())


