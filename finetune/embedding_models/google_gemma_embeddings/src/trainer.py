"""
Trainer for Google Gemma 300M Embeddings Fine-tuning
"""

import os
import json
import logging
import random
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from datasets import Dataset as HFDataset
import wandb

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation_utils import EmbeddingEvaluator, prepare_evaluation_data
from data_loader import EmbeddingDataLoader as QADataLoader



class EvalCallback(TrainerCallback):
    """Custom evaluation callback for Gemma training"""
    
    def __init__(self, eval_func):
        self.eval_func = eval_func
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if state.global_step % 100 == 0:  # Evaluate every 100 steps
            try:
                eval_results = self.eval_func()
                if eval_results:
                    logs.update(eval_results)
                    print(f"Step {state.global_step}: Evaluation completed")
            except Exception as e:
                print(f"Evaluation failed at step {state.global_step}: {e}")

class GemmaEmbeddingTrainer:
    """Trainer class for Google Gemma 300M embeddings"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Setup directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Setup wandb if enabled
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Load data using EmbeddingDataLoader
        self.data_loader = QADataLoader(self.config.data_path, self.config.test_split)
        self.data_loader.print_data_info()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (create log directory if not exists)
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
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
    
    def _prepare_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and evaluation data"""
        self.logger.info("Loading and preparing data...")
        
        # Use existing data_loader
        train_data = self.data_loader.train_data
        test_data = self.data_loader.test_data
        
        # Log dataset information
        self.logger.info("=" * 50)
        self.logger.info("DATASET INFORMATION")
        self.logger.info("=" * 50)
        
        total_samples = len(train_data) + len(test_data)
        train_samples = len(train_data)
        test_samples = len(test_data)
        
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Train samples: {train_samples}")
        self.logger.info(f"Test samples: {test_samples}")
        
        # Calculate average lengths
        train_questions = [item['question'] for item in train_data]
        train_answers = [item['answer'] for item in train_data]
        
        avg_q_len = np.mean([len(q.split()) for q in train_questions])
        avg_a_len = np.mean([len(a.split()) for a in train_answers])
        
        self.logger.info(f"Average question length: {avg_q_len:.1f} words")
        self.logger.info(f"Average answer length: {avg_a_len:.1f} words")
        
        # Count question types
        question_types = {}
        for item in train_data:
            q_type = item.get('type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        self.logger.info("\nQuestion types:")
        for q_type, count in question_types.items():
            percentage = (count / train_samples) * 100
            self.logger.info(f"  {q_type}: {count} ({percentage:.1f}%)")
        
        self.logger.info("=" * 50)
        
        return train_data, test_data
    
    def _prepare_datasets(self, train_data: List[Dict], eval_data: List[Dict]) -> Tuple[HFDataset, HFDataset]:
        """Convert data to Dataset format for SentenceTransformer training
        
        IMPORTANT: MultipleNegativesRankingLoss only needs positive pairs!
        It automatically treats other samples in the batch as negatives.
        Do NOT manually add negative examples with label=0.0
        """
        # For MultipleNegativesRankingLoss, we only need positive pairs
        # The loss function will automatically use other samples in batch as negatives
        train_examples = []
        for item in train_data:
            # Only add positive pairs - no manual negatives needed
            train_examples.append(InputExample(texts=[item['question'], item['answer']]))
        
        # Create evaluation dataset (only positive examples)
        eval_examples = []
        for item in eval_data:
            eval_examples.append(InputExample(texts=[item['question'], item['answer']]))
        
        self.logger.info(f"Training examples: {len(train_examples)} (positive pairs only)")
        self.logger.info(f"Evaluation examples: {len(eval_examples)} (positive pairs only)")
        self.logger.info("MultipleNegativesRankingLoss will automatically use in-batch negatives")
        
        # Convert InputExample lists to Dataset format
        train_dataset = HFDataset.from_list([
            {
                'sentence_0': example.texts[0],
                'sentence_1': example.texts[1]
                # No label needed for MultipleNegativesRankingLoss
            } for example in train_examples
        ])
        
        eval_dataset = HFDataset.from_list([
            {
                'sentence_0': example.texts[0],
                'sentence_1': example.texts[1]
                # No label needed for evaluation either
            } for example in eval_examples
        ])
        
        return train_dataset, eval_dataset
    
    def _load_model(self):
        """Load the Google Gemma embedding model"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model
        self.model = SentenceTransformer(self.config.model_name)
        
        # Set max sequence length if specified
        if hasattr(self.config, 'max_seq_length'):
            self.model.max_seq_length = self.config.max_seq_length
        
        self.logger.info("Model loaded successfully")
        return self.model
    
    def _create_loss_function(self):
        """Create loss function for training"""
        return MultipleNegativesRankingLoss(self.model)
    
    def _create_evaluation_function(self, eval_dataset):
        """Create comprehensive evaluation function with detailed metrics"""
        def evaluate():
            try:
                self.logger.info("Running comprehensive evaluation...")
                
                # Get evaluation data
                eval_data = eval_dataset.to_list() if hasattr(eval_dataset, 'to_list') else list(eval_dataset)
                eval_sample = eval_data[:100]  # Use first 100 samples for evaluation
                
                queries = [item['sentence_0'] for item in eval_sample]
                positives = [item['sentence_1'] for item in eval_sample]
                
                # Encode queries and positives
                query_embeddings = self.model.encode(queries, convert_to_tensor=True)
                positive_embeddings = self.model.encode(positives, convert_to_tensor=True)
                
                # Calculate cosine similarities
                similarities = torch.nn.functional.cosine_similarity(
                    query_embeddings, positive_embeddings, dim=1
                ).cpu().numpy()
                
                # Basic similarity metrics
                mean_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                min_similarity = np.min(similarities)
                max_similarity = np.max(similarities)
                
                # Accuracy with different thresholds
                accuracy_05 = np.mean(similarities > 0.5)
                accuracy_06 = np.mean(similarities > 0.6)
                accuracy_07 = np.mean(similarities > 0.7)
                
                # Prepare data for advanced metrics using EmbeddingEvaluator
                evaluator = EmbeddingEvaluator(k_values=[1, 3, 5, 10])
                
                # For retrieval metrics, we need to simulate a retrieval scenario
                # Create a corpus from all positive answers
                corpus = positives.copy()
                
                # Calculate similarities between queries and all corpus items
                corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
                
                relevant_docs = []
                retrieved_docs = []
                relevance_scores = []
                
                for i, query_emb in enumerate(query_embeddings):
                    # Calculate similarities with all corpus items
                    query_corpus_sims = torch.nn.functional.cosine_similarity(
                        query_emb.unsqueeze(0), corpus_embeddings, dim=1
                    ).cpu().numpy()
                    
                    # Sort by similarity (descending)
                    sorted_indices = np.argsort(query_corpus_sims)[::-1]
                    
                    # The relevant document is the positive answer (index i)
                    relevant_docs.append([i])
                    retrieved_docs.append(sorted_indices.tolist())
                    relevance_scores.append([1.0])  # Binary relevance
                
                # Calculate advanced metrics
                metrics_results = evaluator.evaluate_all_metrics(
                    relevant_docs, retrieved_docs, relevance_scores
                )
                
                # Compile all metrics
                eval_metrics = {
                    'eval/mean_similarity': mean_similarity,
                    'eval/std_similarity': std_similarity,
                    'eval/min_similarity': min_similarity,
                    'eval/max_similarity': max_similarity,
                    'eval/accuracy_0.5': accuracy_05,
                    'eval/accuracy_0.6': accuracy_06,
                    'eval/accuracy_0.7': accuracy_07,
                    'eval/num_samples': len(similarities),
                }
                
                # Add advanced metrics
                for metric_name, k_results in metrics_results.items():
                    for k, value in k_results.items():
                        eval_metrics[f'eval/{metric_name}@{k}'] = value
                
                # Log to wandb if enabled
                if self.config.use_wandb and wandb.run is not None:
                    wandb.log({
                        **eval_metrics,
                        'eval/similarity_distribution': wandb.Histogram(similarities)
                    })
                
                # Log summary to console
                self.logger.info("=" * 60)
                self.logger.info("EVALUATION RESULTS")
                self.logger.info("=" * 60)
                self.logger.info(f"Mean Similarity: {mean_similarity:.4f} ± {std_similarity:.4f}")
                self.logger.info(f"Similarity Range: [{min_similarity:.4f}, {max_similarity:.4f}]")
                self.logger.info(f"Accuracy@0.5: {accuracy_05:.4f}")
                self.logger.info(f"Accuracy@0.6: {accuracy_06:.4f}")
                self.logger.info(f"Accuracy@0.7: {accuracy_07:.4f}")
                
                # Log retrieval metrics
                for metric_name, k_results in metrics_results.items():
                    metric_str = f"{metric_name}: "
                    metric_values = [f"{metric_name}@{k}={v:.4f}" for k, v in k_results.items()]
                    self.logger.info(metric_str + ", ".join(metric_values))
                
                self.logger.info("=" * 60)
                
                return eval_metrics
                
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {}
        
        return evaluate
    
    def train(self):
        """Main training function"""
        self.logger.info("Starting Gemma embedding training...")
        
        try:
            # Prepare data
            train_data, eval_data = self._prepare_data()
            
            # Load model
            self.model = self._load_model()
            
            # Prepare datasets
            train_dataset, eval_dataset = self._prepare_datasets(train_data, eval_data)
            
            # Create loss function
            loss_function = self._create_loss_function()
            
            # Create evaluation function
            eval_func = self._create_evaluation_function(eval_dataset)
            
            # Calculate warmup steps based on actual dataset size
            warmup_steps = self.config.calculate_warmup_steps(len(train_dataset))
            self.logger.info(f"Calculated warmup steps: {warmup_steps} (ratio: {self.config.warmup_ratio})")
            
            # Setup training arguments
            training_args = SentenceTransformerTrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=getattr(self.config, 'gradient_accumulation_steps', 1),
                learning_rate=self.config.learning_rate,
                warmup_steps=warmup_steps,  # Use calculated warmup_steps
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', False),
                dataloader_drop_last=False,
                dataloader_num_workers=0,
                report_to="wandb" if self.config.use_wandb else None,
                run_name=self.config.wandb_run_name,
            )
            
            # Add gradient clipping
            training_args.max_grad_norm = self.config.max_grad_norm
            
            # Add cosine annealing scheduler
            training_args.lr_scheduler_type = "cosine"
            
            # Create trainer
            trainer = SentenceTransformerTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss_function,
                callbacks=[EvalCallback(eval_func)]
            )
            
            # Start training
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save final model
            final_model_path = os.path.join(self.config.output_dir, "final_model")
            self.model.save(final_model_path)
            self.logger.info(f"Final model saved to: {final_model_path}")
            
            # Final evaluation
            self._final_evaluation(eval_dataset)
            
            self.logger.info("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.config.use_wandb:
                wandb.finish()
    
    def _final_evaluation(self, eval_dataset):
        """Run final comprehensive evaluation with all metrics"""
        self.logger.info("Running final comprehensive evaluation...")
        
        try:
            # Use the same comprehensive evaluation function
            eval_func = self._create_evaluation_function(eval_dataset)
            final_results = eval_func()
            
            if final_results:
                self.logger.info("=" * 80)
                self.logger.info("FINAL EVALUATION RESULTS")
                self.logger.info("=" * 80)
                
                # Log all metrics
                for metric_name, value in final_results.items():
                    if isinstance(value, float):
                        self.logger.info(f"{metric_name}: {value:.4f}")
                    else:
                        self.logger.info(f"{metric_name}: {value}")
                
                # Log to wandb if enabled
                if self.config.use_wandb and wandb.run is not None:
                    final_metrics = {f"final_{k}": v for k, v in final_results.items()}
                    wandb.log(final_metrics)
                
                self.logger.info("=" * 80)
                
                # Return mean similarity as main metric
                return final_results.get('eval/mean_similarity', 0.0)
            else:
                self.logger.warning("Final evaluation returned no results")
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model:
            self.model.save(path)
            self.logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = SentenceTransformer(path)
        self.logger.info(f"Model loaded from: {path}")
