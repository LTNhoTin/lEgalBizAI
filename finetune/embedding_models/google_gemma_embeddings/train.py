"""
Training script for Google Gemma 300M Embeddings Fine-tuning
"""

import os
import sys
import logging
import yaml
from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from trainer import GemmaEmbeddingTrainer

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

@dataclass
class GemmaEmbeddingConfig:
    """Configuration for Google Gemma 300M embedding fine-tuning"""
    
    def __init__(self):
        # Load config from YAML
        yaml_config = load_config()
        gemma_config = yaml_config['models']['gemma']
        general_config = yaml_config['general']
        
        # Model configuration
        self.model_name = gemma_config['model_name']
        self.max_seq_length = gemma_config['max_seq_length']
        
        # Training configuration
        self.batch_size = gemma_config['batch_size']
        self.gradient_accumulation_steps = gemma_config['gradient_accumulation_steps']
        self.learning_rate = gemma_config['learning_rate']
        self.num_epochs = gemma_config['num_epochs']
        self.weight_decay = gemma_config['weight_decay']
        self.max_grad_norm = gemma_config['max_grad_norm']
        
        # Warmup ratio (will calculate actual steps later)
        self.warmup_ratio = gemma_config.get('warmup_ratio', 0.1)
        
        # Logging and evaluation
        self.logging_steps = gemma_config['logging_steps']
        self.eval_steps = gemma_config['eval_steps']
        self.save_steps = gemma_config['save_steps']
        # Precision settings - use Gemma-specific config if available, otherwise general
        self.fp16 = gemma_config.get('use_fp16', general_config['use_fp16'])
        self.bf16 = gemma_config.get('use_bf16', general_config['use_bf16'])
        self.gradient_checkpointing = False  # Keep False for small model + small batch
        
        # Evaluation configuration
        self.k_values = [1, 3, 5, 10]
        
        # Wandb configuration
        self.use_wandb = general_config['use_wandb']
        self.wandb_project = general_config['wandb_project']
        self.wandb_run_name = gemma_config['wandb_run_name']
        
        # Data configuration
        self.data_path = general_config['data_path']
        self.test_split = general_config['test_split']
        self.num_negatives = gemma_config['num_negatives']
        
        # Output configuration
        self.output_dir = gemma_config['output_dir']
        self.log_dir = "./logs"
        self.log_level = "INFO"
    
    def calculate_warmup_steps(self, dataset_size: int) -> int:
        """Calculate warmup steps based on dataset size and training parameters"""
        steps_per_epoch = dataset_size // (self.batch_size * self.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        return max(1, warmup_steps)  # Ensure at least 1 warmup step

def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = GemmaEmbeddingConfig()
    
    logger.info("Starting Google Gemma 300M embedding fine-tuning")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of epochs: {config.num_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    try:
        # Initialize trainer
        trainer = GemmaEmbeddingTrainer(config)
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()