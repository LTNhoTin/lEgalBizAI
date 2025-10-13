"""
All-MiniLM-L6-v2 Training Script - Optimized
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
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_loader import EmbeddingDataLoader

# Import our trainer
from src.trainer import MiniLMTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MiniLMConfig:
    """Configuration for all-MiniLM-L6-v2 embedding model finetuning
    All values are loaded from config.yaml - no hardcoded defaults
    """
    
    # Model configuration (loaded from config.yaml)
    model_name: str
    max_seq_length: int
    embedding_dim: int
    
    # Training configuration (loaded from config.yaml)
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_accumulation_steps: int
    
    # Mixed precision (loaded from config.yaml)
    use_fp16: bool
    use_bf16: bool
    
    # Data configuration (loaded from config.yaml)
    data_path: str
    test_split: float
    negative_samples: int
    
    # Evaluation configuration (loaded from config.yaml)
    eval_steps: int
    save_steps: int
    logging_steps: int
    
    # Hardware configuration (loaded from config.yaml)
    device: str
    
    # Contrastive learning (loaded from config.yaml)
    temperature: float
    
    # Wandb configuration (loaded from config.yaml)
    use_wandb: bool
    wandb_project: str
    wandb_entity: str
    wandb_run_name: str
    
    # Output configuration (loaded from config.yaml)
    output_dir: str
    
    # Fields with default values (must be at the end)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    k_values: List[int] = None
    scale: float = 20.0  # Keep this as it's not in config.yaml
    loss_function: str = "MultipleNegativesRankingLoss"  # Keep this as it's not in config.yaml
    log_dir: str = "./logs"  # Keep this as it's not in config.yaml
    log_level: str = "INFO"  # Keep this as it's not in config.yaml
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]


class MiniLMDataset(Dataset):
    """Simple dataset wrapper for MiniLM training"""
    
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


def create_minilm_config(yaml_config: Dict[str, Any]) -> MiniLMConfig:
    """Create MiniLM training config from YAML configuration"""
    
    # Get general and MiniLM-specific configs
    general_config = yaml_config.get('general', {})
    minilm_config = yaml_config.get('models', {}).get('minilm', {})
    
    # Merge configs with MiniLM-specific taking precedence
    merged_config = {**general_config, **minilm_config}
    
    # Create MiniLM training config
    config = MiniLMConfig(
        # Model settings
        model_name=merged_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        max_seq_length=merged_config.get('max_seq_length', 256),
        embedding_dim=merged_config.get('embedding_dim', 384),
        
        # Training settings
        batch_size=merged_config.get('batch_size', 32),
        num_epochs=merged_config.get('num_epochs', 30),
        learning_rate=merged_config.get('learning_rate', 0.00002),
        weight_decay=merged_config.get('weight_decay', 0.01),
        warmup_steps=merged_config.get('warmup_steps', 500),
        gradient_accumulation_steps=merged_config.get('gradient_accumulation_steps', 4),
        
        # Mixed precision - prefer MiniLM config over general
        use_fp16=merged_config.get('use_fp16', general_config.get('use_fp16', True)),
        use_bf16=merged_config.get('use_bf16', general_config.get('use_bf16', False)),
        
        # Data configuration
        data_path=general_config.get('data_path', ''),
        test_split=general_config.get('test_split', 0.2),
        negative_samples=merged_config.get('negative_samples', 5),
        
        # Logging & checkpoints
        logging_steps=merged_config.get('logging_steps', 100),
        eval_steps=merged_config.get('eval_steps', 1000),
        save_steps=merged_config.get('save_steps', 1000),
        output_dir=merged_config.get('output_dir', './output/minilm'),
        
        # Evaluation
        k_values=general_config.get('k_values', [1, 3, 5, 10]),
        
        # Hardware
        device=general_config.get('device', 'auto'),
        
        # Contrastive learning
        temperature=merged_config.get('temperature', 0.07),
        
        # Wandb
        use_wandb=general_config.get('use_wandb', True),
        wandb_project=general_config.get('wandb_project', 'legal-embedding-models'),
        wandb_entity=general_config.get('wandb_entity', ''),
        wandb_run_name=merged_config.get('wandb_run_name', 'minilm-l6-v2-embedding-finetune')
    )
    
    return config


def prepare_datasets(config: MiniLMConfig, yaml_config: Dict[str, Any]) -> tuple:
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
    num_negatives = config.negative_samples
    
    train_contrastive = data_loader.create_contrastive_pairs(
        data_loader.train_data, num_negatives
    )
    test_contrastive = data_loader.create_contrastive_pairs(
        data_loader.test_data, num_negatives
    )
    
    logger.info(f"Created {len(train_contrastive)} training pairs")
    logger.info(f"Created {len(test_contrastive)} evaluation pairs")
    
    # Create datasets
    train_dataset = MiniLMDataset(train_contrastive)
    eval_dataset = MiniLMDataset(test_contrastive)
    
    return train_dataset, eval_dataset


def save_config(config: MiniLMConfig, output_dir: str):
    """Save training configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training config saved to: {config_path}")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train all-MiniLM-L6-v2 embedding model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../../config.yaml",
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
    
    # Setup logging
    setup_logging()
    
    # Load YAML configuration
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    yaml_config = load_config_from_yaml(config_path)
    
    # Check if MiniLM is enabled
    if not yaml_config.get('models', {}).get('minilm', {}).get('enabled', False):
        logger.warning("MiniLM is not enabled in config.yaml")
        logger.info("Set models.minilm.enabled: true to enable training")
        sys.exit(0)
    
    # Create MiniLM training config
    config = create_minilm_config(yaml_config)
    
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
    logger.info("=== All-MiniLM-L6-v2 Training Configuration ===")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Embedding dimension: {config.embedding_dim}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Number of epochs: {config.num_epochs}")
    logger.info(f"Warmup steps: {config.warmup_steps}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Mixed precision: FP16={config.use_fp16}, BF16={config.use_bf16}")
    logger.info(f"Negative samples: {config.negative_samples}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Wandb enabled: {config.use_wandb}")
    if config.use_wandb:
        logger.info(f"  Project: {config.wandb_project}")
        logger.info(f"  Run name: {config.wandb_run_name}")
    
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
        trainer = MiniLMTrainer(config)
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        sys.exit(1)
    
    # Start training
    try:
        logger.info("Starting all-MiniLM-L6-v2 training...")
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