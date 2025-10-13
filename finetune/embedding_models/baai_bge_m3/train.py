"""
BGE-M3 Training Script - Optimized
Loads configuration from config.yaml and runs training with optimized trainer
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import EmbeddingDataLoader

# Import our optimized trainer
from src.trainer import BGETrainingConfig, OptimizedBGETrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEDataset(Dataset):
    """Simple dataset wrapper for BGE training"""
    
    def __init__(self, contrastive_pairs: List[Dict[str, Any]]):
        self.pairs = contrastive_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'query': pair['query'],
            'positive': pair['positive'],
            'negative': pair['negatives'][0] if pair['negatives'] else pair['positive']  # Fallback
        }


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def create_bge_config(yaml_config: Dict[str, Any]) -> BGETrainingConfig:
    """Create BGE training config from YAML configuration"""
    
    # Get general and BGE-specific configs
    general_config = yaml_config.get('general', {})
    bge_config = yaml_config.get('models', {}).get('bge_m3', {})
    
    # Merge configs with BGE-specific taking precedence
    merged_config = {**general_config, **bge_config}
    
    # Create BGE training config
    config = BGETrainingConfig(
        # Model settings
        model_name=merged_config.get('model_name', 'BAAI/bge-m3'),
        max_seq_length=merged_config.get('max_seq_length', 384),
        
        # Training settings
        batch_size=merged_config.get('batch_size', 2),
        num_epochs=merged_config.get('num_epochs', 30),
        learning_rate=merged_config.get('learning_rate', 1e-5),
        weight_decay=merged_config.get('weight_decay', 0.01),
        warmup_ratio=merged_config.get('warmup_ratio', 0.1),
        gradient_accumulation_steps=merged_config.get('gradient_accumulation_steps', 2),
        max_grad_norm=merged_config.get('max_grad_norm', 1.0),
        
        # LoRA settings
        use_lora=merged_config.get('use_lora', True),
        lora_r=merged_config.get('lora_r', 8),
        lora_alpha=merged_config.get('lora_alpha', 16),
        lora_dropout=merged_config.get('lora_dropout', 0.1),
        
        # BGE-M3 specific
        use_dense=merged_config.get('use_dense', True),
        use_sparse=merged_config.get('use_sparse', False),
        use_colbert=merged_config.get('use_colbert', False),
        
        # Mixed precision - prefer BGE config over general
        use_fp16=merged_config.get('use_fp16', general_config.get('use_fp16', True)),
        use_bf16=merged_config.get('use_bf16', general_config.get('use_bf16', False)),
        
        # Logging & checkpoints
        logging_steps=merged_config.get('logging_steps', 100),
        eval_steps=merged_config.get('eval_steps', 1000),
        save_steps=merged_config.get('save_steps', 1000),
        output_dir=merged_config.get('output_dir', './output/bge_m3'),
        
        # Wandb
        use_wandb=general_config.get('use_wandb', True),
        wandb_project=general_config.get('wandb_project', 'legal-embedding-models'),
        wandb_run_name=merged_config.get('wandb_run_name', 'bge-m3-optimized')
    )
    
    return config


def prepare_datasets(config: BGETrainingConfig, yaml_config: Dict[str, Any]) -> tuple:
    """Prepare training and evaluation datasets"""
    
    # Get data path from general config
    data_path = yaml_config.get('general', {}).get('data_path')
    test_split = yaml_config.get('general', {}).get('test_split', 0.2)
    
    if not data_path:
        raise ValueError("data_path not found in general config")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    # Load data using existing data loader
    data_loader = EmbeddingDataLoader(data_path, test_split)
    data_loader.print_data_info()
    
    # Create contrastive pairs
    num_negatives = 1  # Simplified to 1 negative per query for efficiency
    
    train_contrastive = data_loader.create_contrastive_pairs(
        data_loader.train_data, num_negatives
    )
    test_contrastive = data_loader.create_contrastive_pairs(
        data_loader.test_data, num_negatives
    )
    
    logger.info(f"Created {len(train_contrastive)} training pairs")
    logger.info(f"Created {len(test_contrastive)} evaluation pairs")
    
    # Create datasets
    train_dataset = BGEDataset(train_contrastive)
    eval_dataset = BGEDataset(test_contrastive)
    
    return train_dataset, eval_dataset


def save_config(config: BGETrainingConfig, output_dir: str):
    """Save training configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training config saved to: {config_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train BGE-M3 embedding model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../config.yaml",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Load YAML configuration
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    yaml_config = load_config_from_yaml(config_path)
    
    # Check if BGE-M3 is enabled
    if not yaml_config.get('models', {}).get('bge_m3', {}).get('enabled', False):
        logger.warning("BGE-M3 is not enabled in config.yaml")
        logger.info("Set models.bge_m3.enabled: true to enable training")
        sys.exit(0)
    
    # Create BGE training config
    config = create_bge_config(yaml_config)
    
    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.use_wandb:
        config.use_wandb = True
    if args.no_wandb:
        config.use_wandb = False
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, config.output_dir)
    
    # Log configuration
    logger.info("=== BGE-M3 Training Configuration ===")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Number of epochs: {config.num_epochs}")
    logger.info(f"LoRA enabled: {config.use_lora}")
    if config.use_lora:
        logger.info(f"  LoRA r: {config.lora_r}")
        logger.info(f"  LoRA alpha: {config.lora_alpha}")
        logger.info(f"  LoRA dropout: {config.lora_dropout}")
    logger.info(f"Mixed precision: FP16={config.use_fp16}, BF16={config.use_bf16}")
    logger.info(f"BGE features: Dense={config.use_dense}, Sparse={config.use_sparse}, ColBERT={config.use_colbert}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Wandb enabled: {config.use_wandb}")
    
    # Prepare datasets
    try:
        train_dataset, eval_dataset = prepare_datasets(config, yaml_config)
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, training will be slow on CPU")
    
    # Initialize trainer
    try:
        trainer = OptimizedBGETrainer(config, train_dataset, eval_dataset)
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        sys.exit(1)
    
    # Start training
    try:
        logger.info("Starting BGE-M3 training...")
        trainer.train()
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()