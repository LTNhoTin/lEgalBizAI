#!/usr/bin/env python3
"""
MiniLM-L6-v2 Fine-tuning Script
Loads configuration from config.yaml and trains the model
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trainer import MiniLMTrainer

@dataclass
class MiniLMConfig:
    """Configuration class for MiniLM training"""
    # Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    embedding_dim: int = 384
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    
    # Mixed precision
    use_fp16: bool = True
    use_bf16: bool = False
    
    # Data settings
    data_path: str = ""
    test_split: float = 0.2
    negative_samples: int = 5
    
    # Contrastive learning
    temperature: float = 0.07
    
    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    output_dir: str = "./output/minilm"
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "legal-embedding-models"
    wandb_entity: str = "nhotin911"
    wandb_run_name: str = "minilm-l6-v2-embedding-finetune"
    
    # Evaluation
    k_values: list = None
    
    # Device settings
    device: str = "auto"
    seed: int = 42

def load_config(config_path: str = None) -> MiniLMConfig:
    """Load configuration from YAML file"""
    if config_path is None:
        # Default config path - adjust based on your project structure
        config_path = "/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # Extract general and minilm specific configs
    general_config = yaml_config.get('general', {})
    minilm_config = yaml_config.get('models', {}).get('minilm', {})
    
    # Create config object
    config = MiniLMConfig()
    
    # Update from general config
    config.data_path = general_config.get('data_path', config.data_path)
    config.test_split = general_config.get('test_split', config.test_split)
    config.use_wandb = general_config.get('use_wandb', config.use_wandb)
    config.wandb_project = general_config.get('wandb_project', config.wandb_project)
    config.wandb_entity = general_config.get('wandb_entity', config.wandb_entity)
    config.device = general_config.get('device', config.device)
    config.seed = general_config.get('seed', config.seed)
    config.k_values = general_config.get('k_values', [1, 3, 5, 10])
    
    # Override with minilm specific config
    for key, value in minilm_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('minilm_training.log')
        ]
    )

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MiniLM-L6-v2 embedding model')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases')
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Override config with command line arguments
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
        if args.device:
            config.device = args.device
        
        # Log configuration
        logger.info("Training Configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Data: {config.data_path}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  Output: {config.output_dir}")
        logger.info(f"  Use W&B: {config.use_wandb}")
        
        # Create trainer and start training
        trainer = MiniLMTrainer(config)
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()