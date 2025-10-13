"""
MiniLM-L6-v2 Trainer
Implements training and evaluation for sentence-transformers MiniLM model
"""

import os
import json
import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import wandb

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation_utils import EmbeddingEvaluator
from data_loader import EmbeddingDataLoader as QADataLoader

class MiniLMDataset:
    """Dataset class for MiniLM training"""
    
    def __init__(self, data: List[Dict], negative_samples: int = 5):
        self.data = data
        self.negative_samples = negative_samples
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[InputExample]:
        """Create InputExample objects for sentence-transformers"""
        examples = []
        
        # Create positive pairs
        for item in self.data:
            question = item['question']
            answer = item['answer']
            # Create positive example with high similarity score
            examples.append(InputExample(texts=[question, answer], label=1.0))
        
        # Create negative pairs for contrastive learning
        all_answers = [item['answer'] for item in self.data]
        
        for item in self.data:
            question = item['question']
            positive_answer = item['answer']
            
            # Get random negative answers
            negative_answers = [ans for ans in all_answers if ans != positive_answer]
            selected_negatives = random.sample(
                negative_answers, 
                min(self.negative_samples, len(negative_answers))
            )
            
            # Create negative examples
            for neg_answer in selected_negatives:
                examples.append(InputExample(texts=[question, neg_answer], label=0.0))
        
        return examples

class MiniLMTrainer:
    """Trainer class for MiniLM model"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Setup device
        self.device = self._setup_device()
        
        # Load data
        self.data_loader = QADataLoader(config.data_path, config.test_split)
        self.train_data, self.test_data = self._prepare_data()
        
        # Initialize model
        self.model = self._load_model()
        
        # Setup evaluation
        self.evaluator = EmbeddingEvaluator(k_values=config.k_values)
        
        # Setup Weights & Biases
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"MiniLMTrainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _setup_device(self) -> str:
        """Setup device for training"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    torch.cuda.empty_cache()
                    test_tensor = torch.tensor([1.0]).cuda()
                    device = "cuda"
                    self.logger.info("CUDA is available and working")
                except Exception as e:
                    self.logger.warning(f"CUDA available but not working: {e}")
                    device = "cpu"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
                reinit=True
            )
            self.logger.info("Wandb initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False
    
    def _prepare_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and test data"""
        self.logger.info("Loading and preparing data...")
        
        train_data = self.data_loader.train_data
        test_data = self.data_loader.test_data
        
        self.logger.info(f"Train: {len(train_data)}, Test: {len(test_data)} QA pairs")
        
        return train_data, test_data
    
    def _load_model(self) -> SentenceTransformer:
        """Load and configure the SentenceTransformer model"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        model = SentenceTransformer(self.config.model_name, device=self.device)
        
        # Set max sequence length
        model.max_seq_length = self.config.max_seq_length
        
        self.logger.info(f"Model loaded with max_seq_length: {self.config.max_seq_length}")
        
        return model
    
    def _create_loss_function(self):
        """Create loss function for training"""
        # Use CosineSimilarityLoss for similarity learning
        return losses.CosineSimilarityLoss(model=self.model)
    
    def _create_evaluator(self) -> evaluation.SentenceEvaluator:
        """Create evaluator for validation during training"""
        # Create evaluation examples from test data
        eval_examples = []
        for item in self.test_data[:100]:  # Use subset for faster evaluation
            question = item['question']
            answer = item['answer']
            eval_examples.append(InputExample(texts=[question, answer], label=1.0))
        
        # Create similarity evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples, 
            name='test_evaluation'
        )
        
        return evaluator
    
    def train(self):
        """Main training function"""
        self.logger.info("Starting training...")
        
        # Create training dataset
        train_dataset = MiniLMDataset(
            self.train_data, 
            negative_samples=self.config.negative_samples
        )
        train_examples = train_dataset.examples
        
        self.logger.info(f"Created {len(train_examples)} training examples")
        
        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.config.batch_size
        )
        
        # Create loss function
        train_loss = self._create_loss_function()
        
        # Create evaluator
        evaluator = self._create_evaluator()
        
        # Training arguments
        warmup_steps = min(self.config.warmup_steps, len(train_dataloader) // 4)
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Epochs: {self.config.num_epochs}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Warmup steps: {warmup_steps}")
        self.logger.info(f"  Training examples: {len(train_examples)}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup W&B callback for logging
        class WandbCallback:
            def __init__(self, config, use_wandb, logger):
                self.config = config
                self.use_wandb = use_wandb
                self.logger = logger
                self.step = 0
            
            def __call__(self, score, epoch, steps):
                """Called after each evaluation"""
                self.step += 1
                
                if self.use_wandb:
                    try:
                        wandb.log({
                            'eval/similarity_score': score,
                            'eval/epoch': epoch,
                            'eval/steps': steps,
                            'train/global_step': self.step
                        })
                    except Exception as e:
                        self.logger.warning(f"W&B logging failed: {e}")
                
                self.logger.info(f"Epoch {epoch}, Steps {steps}, Score: {score:.4f}")
        
        callback = WandbCallback(self.config, self.config.use_wandb, self.logger)
        
        # Train the model
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.config.num_epochs,
                evaluation_steps=self.config.eval_steps,
                warmup_steps=warmup_steps,
                output_path=self.config.output_dir,
                save_best_model=True,
                optimizer_params={'lr': self.config.learning_rate},
                weight_decay=self.config.weight_decay,
                use_amp=self.config.use_fp16,
                callback=callback
            )
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        self.logger.info("Training completed!")
        
        # Final evaluation
        self._final_evaluation()
    
    def _final_evaluation(self):
        """Perform comprehensive final evaluation"""
        self.logger.info("Performing final evaluation...")
        
        # Encode all queries and passages
        queries = [item['question'] for item in self.test_data]
        passages = [item['answer'] for item in self.test_data]
        
        self.logger.info(f"Encoding {len(queries)} queries and {len(passages)} passages...")
        
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        passage_embeddings = self.model.encode(passages, convert_to_tensor=True)
        
        # Calculate similarity matrix
        similarity_matrix = torch.nn.functional.cosine_similarity(
            query_embeddings.unsqueeze(1), 
            passage_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Calculate basic similarity metrics
        diagonal_similarities = torch.diagonal(similarity_matrix).cpu().numpy()
        mean_similarity = np.mean(diagonal_similarities)
        std_similarity = np.std(diagonal_similarities)
        
        # Calculate retrieval metrics
        relevant_docs = []
        retrieved_docs = []
        
        for i in range(len(queries)):
            # For each query, the relevant document is its corresponding passage (index i)
            relevant_docs.append([i])
            
            # Get similarity scores for this query with all passages
            similarities = similarity_matrix[i].cpu().numpy()
            
            # Get indices sorted by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            retrieved_docs.append(sorted_indices.tolist())
        
        # Calculate metrics using evaluator
        metrics = self.evaluator.evaluate_all_metrics(relevant_docs, retrieved_docs)
        
        # Compile all metrics
        eval_metrics = {
            'eval/mean_similarity': mean_similarity,
            'eval/std_similarity': std_similarity,
            'eval/min_similarity': np.min(diagonal_similarities),
            'eval/max_similarity': np.max(diagonal_similarities),
            'eval/num_samples': len(self.test_data),
        }
        
        # Add accuracy metrics
        for threshold in [0.5, 0.6, 0.7]:
            accuracy = np.mean(diagonal_similarities > threshold)
            eval_metrics[f'eval/accuracy_{threshold}'] = accuracy
        
        # Add advanced metrics
        for metric_name, k_results in metrics.items():
            for k, value in k_results.items():
                eval_metrics[f'eval/{metric_name}@{k}'] = value
        
        # Log to wandb if enabled
        if self.config.use_wandb and wandb.run is not None:
            # Add final_ prefix to all metrics for consistency with Gemma
            final_metrics = {f"final_{k.replace('eval/', '')}": v for k, v in eval_metrics.items()}
            final_metrics['final_similarity_distribution'] = wandb.Histogram(diagonal_similarities)
            wandb.log(final_metrics)
        
        # Log summary to console
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Mean Similarity: {mean_similarity:.4f} ± {std_similarity:.4f}")
        self.logger.info(f"Similarity Range: [{np.min(diagonal_similarities):.4f}, {np.max(diagonal_similarities):.4f}]")
        self.logger.info(f"Accuracy@0.5: {eval_metrics['eval/accuracy_0.5']:.4f}")
        self.logger.info(f"Accuracy@0.6: {eval_metrics['eval/accuracy_0.6']:.4f}")
        self.logger.info(f"Accuracy@0.7: {eval_metrics['eval/accuracy_0.7']:.4f}")
        
        # Log retrieval metrics
        for metric_name, k_results in metrics.items():
            metric_str = f"{metric_name}: "
            metric_values = [f"{metric_name}@{k}={v:.4f}" for k, v in k_results.items()]
            self.logger.info(metric_str + ", ".join(metric_values))
        
        self.logger.info("=" * 60)
        
        # Check target metrics
        self._check_target_metrics(metrics)
        
        self.logger.info("Final evaluation completed!")
    
    def _check_target_metrics(self, metrics: Dict):
        """Check if target metrics are achieved"""
        import yaml
        
        # Load target metrics from config.yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        targets = config_data.get('targets', {})
        
        self.logger.info("Checking target metrics...")
        targets_met = True
        
        for target_name, target_value in targets.items():
            if target_name.startswith('recall_at_'):
                k = int(target_name.split('_')[-1])
                actual_value = metrics.get('recall', {}).get(k, 0.0)
            elif target_name.startswith('mrr_at_'):
                k = int(target_name.split('_')[-1])
                actual_value = metrics.get('mrr', {}).get(k, 0.0)
            elif target_name.startswith('ndcg_at_'):
                k = int(target_name.split('_')[-1])
                actual_value = metrics.get('ndcg', {}).get(k, 0.0)
            elif target_name == 'context_precision':
                actual_value = metrics.get('context_precision', {}).get(10, 0.0)
            else:
                continue
            
            met = actual_value >= target_value
            targets_met = targets_met and met
            
            status = "✓" if met else "✗"
            self.logger.info(f"  {status} {target_name}: {actual_value:.4f} (target: {target_value:.4f})")
        
        if targets_met:
            self.logger.info("🎉 All target metrics achieved!")
        else:
            self.logger.warning("⚠️  Some target metrics not achieved")
        
        # Log targets met status to wandb
        if self.config.use_wandb and wandb.run is not None:
            wandb.log({'final_targets_met': targets_met})
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save(path)
        self.logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = SentenceTransformer(path, device=self.device)
        self.logger.info(f"Model loaded from: {path}")