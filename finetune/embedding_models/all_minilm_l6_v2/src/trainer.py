"""
Trainer for all-MiniLM-L6-v2 Embeddings Fine-tuning
"""

import os
import json
import logging
import random
from typing import List, Dict, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
import wandb

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation_utils import EmbeddingEvaluator, prepare_evaluation_data
from data_loader import EmbeddingDataLoader as QADataLoader

class MiniLMDataset:
    """Dataset class for MiniLM training with sentence-transformers"""
    
    def __init__(self, data: List[Dict], loss_function: str = "CosineSimilarityLoss", negative_samples: int = 5):
        self.data = data
        self.loss_function = loss_function
        self.negative_samples = negative_samples
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[InputExample]:
        """Create InputExample objects for sentence-transformers based on loss function"""
        examples = []
        
        if self.loss_function == "CosineSimilarityLoss":
            # CosineSimilarityLoss chỉ cần positive pairs với label liên tục
            for item in self.data:
                question = item['question']
                answer = item['answer']
                # Chỉ tạo positive examples với label=1.0 (similarity score)
                examples.append(InputExample(texts=[question, answer], label=1.0))
                
        elif self.loss_function == "MultipleNegativesRankingLoss":
            # MultipleNegativesRankingLoss chỉ cần positive pairs, tự động tạo negatives trong batch
            for item in self.data:
                question = item['question']
                answer = item['answer']
                # Chỉ positive pairs, không cần label
                examples.append(InputExample(texts=[question, answer]))
                
        else:
            # Fallback cho các loss function khác có thể cần negative samples
            for item in self.data:
                question = item['question']
                answer = item['answer']
                
                # Positive example
                examples.append(InputExample(texts=[question, answer], label=1.0))
                
                # Create negative examples nếu cần
                negative_answers = self._get_negative_answers(answer)
                for neg_answer in negative_answers[:self.negative_samples]:
                    examples.append(InputExample(texts=[question, neg_answer], label=0.0))
        
        return examples
    
    def _get_negative_answers(self, positive_answer: str) -> List[str]:
        """Get negative answers for contrastive learning"""
        all_answers = [item['answer'] for item in self.data]
        negative_answers = [ans for ans in all_answers if ans != positive_answer]
        return random.sample(negative_answers, min(len(negative_answers), self.negative_samples * 2))

class MiniLMTrainer:
    """Trainer class for all-MiniLM-L6-v2 fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = SentenceTransformer(config.model_name)
        self.model.max_seq_length = config.max_seq_length
        
        # Setup device
        if torch.cuda.is_available() and config.device == "cuda":
            self.device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        # Load and prepare data
        self.data_loader = QADataLoader(config.data_path)
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        
        # Initialize evaluator
        self.evaluator = EmbeddingEvaluator(config.k_values)
        
        # Setup Weights & Biases
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create file handler
        log_file = os.path.join(self.config.log_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__,
            tags=["minilm", "sentence-transformers", "embedding", "fine-tuning"]
        )
        
        # Log model architecture
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/trainable_percentage": (trainable_params / total_params) * 100,
            "model/base_model": self.config.model_name,
            "model/embedding_dim": self.config.embedding_dim
        })
    
    def _prepare_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Prepare training, validation, and test data"""
        self.logger.info("Loading and preparing data...")
        
        # Use already loaded data from EmbeddingDataLoader
        train_data = self.data_loader.train_data
        test_data = self.data_loader.test_data
        
        # Split train data into train and validation
        val_split = int(len(train_data) * 0.2)  # 20% for validation
        val_data = train_data[:val_split]
        train_data = train_data[val_split:]
        
        self.logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} QA pairs")
        
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _create_loss_function(self):
        """Create loss function based on configuration"""
        if self.config.loss_function == "CosineSimilarityLoss":
            return losses.CosineSimilarityLoss(self.model)
        elif self.config.loss_function == "MultipleNegativesRankingLoss":
            return losses.MultipleNegativesRankingLoss(self.model, scale=self.config.scale)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def _create_evaluator(self) -> evaluation.SentenceEvaluator:
        """Create custom evaluator with W&B logging for intermediate metrics"""
        from sentence_transformers.evaluation import SentenceEvaluator
        import numpy as np
        
        class WandbEvaluator(SentenceEvaluator):
            def __init__(self, val_data, config, logger, use_wandb=True):
                self.val_data = val_data[:100]  # Use subset for faster evaluation
                self.config = config
                self.logger = logger
                self.use_wandb = use_wandb
                self.name = "validation_wandb"
                
            def __call__(self, model, output_path=None, epoch=-1, steps=-1):
                """Evaluate model and log detailed metrics to W&B"""
                try:
                    # Prepare evaluation data
                    queries = []
                    passages = []
                    labels = []
                    
                    for item in self.val_data:
                        queries.append(item['question'])
                        passages.append(item['answer'])
                        labels.append(1.0)
                    
                    # Encode queries and passages
                    query_embeddings = model.encode(queries, convert_to_tensor=True)
                    passage_embeddings = model.encode(passages, convert_to_tensor=True)
                    
                    # Calculate cosine similarities
                    similarities = torch.nn.functional.cosine_similarity(
                        query_embeddings, passage_embeddings, dim=1
                    ).cpu().numpy()
                    
                    # Calculate metrics
                    mean_similarity = np.mean(similarities)
                    std_similarity = np.std(similarities)
                    min_similarity = np.min(similarities)
                    max_similarity = np.max(similarities)
                    
                    # Calculate accuracy (assuming threshold of 0.5)
                    accuracy = np.mean(similarities > 0.5)
                    
                    # Log to W&B if enabled
                    if self.use_wandb and wandb.run is not None:
                        try:
                            metrics = {
                                'eval/mean_similarity': mean_similarity,
                                'eval/std_similarity': std_similarity,
                                'eval/min_similarity': min_similarity,
                                'eval/max_similarity': max_similarity,
                                'eval/accuracy': accuracy,
                                'eval/num_samples': len(similarities),
                            }
                            
                            # Add distribution histogram
                            wandb.log({
                                **metrics,
                                'eval/similarity_distribution': wandb.Histogram(similarities)
                            })
                        except Exception as e:
                            self.logger.error(f"Failed to log evaluation metrics to wandb: {e}")
                            # Continue without wandb logging
                    
                    # Log to console
                    self.logger.info(f"Evaluation - Epoch {epoch}, Steps {steps}")
                    self.logger.info(f"  Mean similarity: {mean_similarity:.4f}")
                    self.logger.info(f"  Accuracy (>0.5): {accuracy:.4f}")
                    
                    return mean_similarity
                    
                except Exception as e:
                    self.logger.error(f"Evaluation error: {e}")
                    return 0.0
        
        return WandbEvaluator(self.val_data, self.config, self.logger, self.config.use_wandb)
    
    def train(self):
        """Main training function"""
        self.logger.info("Starting training...")
        
        # Create training dataset với loss function
        train_dataset = MiniLMDataset(
            self.train_data, 
            loss_function=self.config.loss_function,
            negative_samples=self.config.negative_samples
        )
        train_examples = train_dataset.examples
        
        self.logger.info(f"Created {len(train_examples)} training examples using {self.config.loss_function}")
        
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
        self.logger.info(f"  Loss function: {self.config.loss_function}")
        
        # Create comprehensive W&B callback for detailed logging
        class ComprehensiveWandbCallback:
            def __init__(self, config, use_wandb, logger):
                self.config = config
                self.use_wandb = use_wandb
                self.logger = logger
                self.global_step = 0
                self.best_score = 0.0
                self.epoch_start_time = None
                self.wandb_failed = False  # Track wandb failures
                
            def _safe_wandb_log(self, metrics, step=None):
                """Safely log to wandb with proper error handling"""
                if not self.use_wandb or self.wandb_failed:
                    return
                    
                try:
                    wandb.log(metrics, step=step)
                except Exception as e:
                    if not self.wandb_failed:  # Log error only once
                        self.logger.error(f"Wandb logging failed: {e}")
                        self.logger.warning("Disabling wandb logging for this session")
                        self.wandb_failed = True
                
            def on_epoch_start(self, epoch):
                """Called at the start of each epoch"""
                import time
                self.epoch_start_time = time.time()
                self._safe_wandb_log({
                    'train/epoch_start': epoch,
                    'train/learning_rate': float(self.config.learning_rate),
                }, step=self.global_step)
                    
            def on_step_end(self, loss, learning_rate, gradient_norm=None):
                """Called after each training step"""
                self.global_step += 1
                metrics = {
                    'train/loss': loss,
                    'train/learning_rate': learning_rate,
                    'train/global_step': self.global_step,
                }
                
                if gradient_norm is not None:
                    metrics['train/gradient_norm'] = gradient_norm
                
                # Log GPU memory usage
                try:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                        metrics.update({
                            'system/gpu_memory_allocated_gb': gpu_memory,
                            'system/gpu_memory_cached_gb': gpu_memory_cached,
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to get GPU memory info: {e}")
                
                self._safe_wandb_log(metrics, step=self.global_step)
                    
            def on_evaluation(self, score, epoch, steps):
                """Called during evaluation"""
                # Log evaluation metrics
                eval_metrics = {
                    'eval/score': score if score is not None else 0,
                    'eval/epoch': epoch,
                    'eval/steps': steps,
                    'train/progress': epoch / self.config.num_epochs,
                }
                
                # Track best score
                if score and score > self.best_score:
                    self.best_score = score
                    eval_metrics['eval/best_score'] = self.best_score
                    eval_metrics['eval/improvement'] = True
                else:
                    eval_metrics['eval/improvement'] = False
                
                self._safe_wandb_log(eval_metrics, step=self.global_step)
                    
            def on_epoch_end(self, epoch):
                """Called at the end of each epoch"""
                if self.epoch_start_time:
                    import time
                    epoch_time = time.time() - self.epoch_start_time
                    self._safe_wandb_log({
                        'train/epoch_time_seconds': epoch_time,
                        'train/epoch_end': epoch,
                    }, step=self.global_step)
                    
            def __call__(self, score, epoch, steps):
                """Compatibility with sentence-transformers callback interface"""
                self.on_evaluation(score, epoch, steps)
        
        # Create callback instance
        callback = ComprehensiveWandbCallback(self.config, self.config.use_wandb, self.logger) if self.config.use_wandb else None
        
        # Debug learning rate type
        print(f"Learning rate type: {type(self.config.learning_rate)}, value: {self.config.learning_rate}")
        lr_value = float(self.config.learning_rate)  # Ensure it's float
        print(f"Converted learning rate: {lr_value}")
        
        # Kiểm tra và điều chỉnh fp16 dựa trên batch size để tránh NaN loss
        use_fp16 = self.config.fp16
        if self.config.fp16 and self.config.batch_size < 4:
            self.logger.warning(f"Batch size {self.config.batch_size} quá nhỏ cho fp16, có thể gây NaN loss. Tắt fp16.")
            use_fp16 = False
        elif self.config.fp16:
            self.logger.info(f"Sử dụng fp16 với batch size {self.config.batch_size}")
        
        # Train the model with comprehensive logging
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.config.num_epochs,
                evaluation_steps=self.config.eval_steps,
                warmup_steps=warmup_steps,
                output_path=self.config.output_dir,
                save_best_model=True,
                optimizer_params={'lr': lr_value},
                weight_decay=self.config.weight_decay,
                use_amp=use_fp16,  # Sử dụng fp16 đã được kiểm tra
                callback=callback  # Enable comprehensive W&B logging
            )
        except Exception as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                self.logger.error(f"NaN/Inf loss detected: {e}")
                if use_fp16:
                    self.logger.warning("Thử lại với fp16=False...")
                    self.model.fit(
                        train_objectives=[(train_dataloader, train_loss)],
                        evaluator=evaluator,
                        epochs=self.config.num_epochs,
                        evaluation_steps=self.config.eval_steps,
                        warmup_steps=warmup_steps,
                        output_path=self.config.output_dir,
                        save_best_model=True,
                        optimizer_params={'lr': lr_value},
                        weight_decay=self.config.weight_decay,
                        use_amp=False,  # Tắt fp16
                        callback=callback
                    )
                else:
                    raise e
            else:
                raise e
        
        self.logger.info("Training completed!")
        
        # Final evaluation
        self._final_evaluation()
    
    def _final_evaluation(self):
        """Perform final evaluation on test set"""
        self.logger.info("Performing final evaluation...")
        
        # Encode queries and passages
        queries = [item['question'] for item in self.test_data]
        passages = [item['answer'] for item in self.test_data]
        
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        passage_embeddings = self.model.encode(passages, convert_to_tensor=True)
        
        # Prepare evaluation data with correct parameters
        relevant_docs, retrieved_docs = prepare_evaluation_data(
            self.test_data, 
            passage_embeddings.cpu().numpy(),
            query_embeddings.cpu().numpy()
        )
        
        # Calculate metrics using the evaluator
        metrics = self.evaluator.evaluate_all_metrics(relevant_docs, retrieved_docs)
        
        # Log metrics with comprehensive W&B logging
        self.logger.info("Final Evaluation Results:")
        wandb_metrics = {}
        
        # Calculate additional statistics
        all_similarities = []
        for i in range(len(query_embeddings)):
            similarity = torch.nn.functional.cosine_similarity(
                query_embeddings[i:i+1], passage_embeddings[i:i+1], dim=1
            ).item()
            all_similarities.append(similarity)
        
        # Log detailed metrics
        for metric_name, values in metrics.items():
            for k, value in values.items():
                self.logger.info(f"  {metric_name}@{k}: {value:.4f}")
                
                # Collect metrics for batch logging
                if self.config.use_wandb:
                    wandb_metrics[f"final/{metric_name}@{k}"] = value
        
        # Log to wandb with comprehensive context
        if self.config.use_wandb and wandb_metrics:
            try:
                # Add evaluation summary
                wandb_metrics.update({
                    'final/num_test_samples': len(self.test_data),
                    'final/mean_similarity': np.mean(all_similarities),
                    'final/std_similarity': np.std(all_similarities),
                    'final/min_similarity': np.min(all_similarities),
                    'final/max_similarity': np.max(all_similarities),
                    'final/accuracy_threshold_0.5': np.mean(np.array(all_similarities) > 0.5),
                    'final/accuracy_threshold_0.7': np.mean(np.array(all_similarities) > 0.7),
                    'final/accuracy_threshold_0.9': np.mean(np.array(all_similarities) > 0.9),
                })
                
                # Log best metrics for each metric type
                for metric_name, values in metrics.items():
                    if values:
                        best_value = max(values.values())
                        best_k = max(values.keys(), key=lambda k: values[k])
                        wandb_metrics[f"final/best_{metric_name}"] = best_value
                        wandb_metrics[f"final/best_{metric_name}_k"] = best_k
                
                # Add training configuration summary
                wandb_metrics.update({
                    'final/total_epochs': self.config.num_epochs,
                    'final/batch_size': self.config.batch_size,
                    'final/learning_rate': float(self.config.learning_rate),
                    'final/loss_function': self.config.loss_function,
                    'final/model_name': self.config.model_name,
                })
                
                # Log similarity distribution histogram
                wandb_metrics['final/similarity_distribution'] = wandb.Histogram(all_similarities)
                
                # Batch log all metrics với error handling
                wandb.log(wandb_metrics)
                
                # Create summary table for key metrics
                summary_data = []
                for metric_name, values in metrics.items():
                    for k, value in values.items():
                        summary_data.append([metric_name, k, value])
                
                if summary_data:
                    table = wandb.Table(
                        columns=["Metric", "K", "Value"],
                        data=summary_data
                    )
                    wandb.log({"final/metrics_summary_table": table})
                    
            except Exception as e:
                self.logger.error(f"Failed to log final metrics to wandb: {e}")
                self.logger.warning("Continuing without wandb logging for final metrics")
        
        # Check if targets are met
        self._check_target_metrics(metrics)
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save(final_model_path)
        self.logger.info(f"Final model saved to: {final_model_path}")
    
    def _check_target_metrics(self, metrics: Dict):
        """Check if target metrics are achieved"""
        import yaml
        
        # Load target metrics from config.yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        TARGET_METRICS = config_data.get('targets', {})
        
        self.logger.info("Checking target metrics...")
        targets_met = True
        
        for metric_name, target_value in TARGET_METRICS.items():
            if metric_name == "context_precision":
                # Context Precision is calculated differently
                continue
                
            # Parse metric name (e.g., "recall_at_5" -> "recall", 5)
            if "_at_" in metric_name:
                base_metric, k_str = metric_name.split("_at_")
                k = int(k_str)
                
                # Check if this metric exists in results
                if base_metric in metrics and k in metrics[base_metric]:
                    actual_value = metrics[base_metric][k]
                    
                    if actual_value >= target_value:
                        self.logger.info(f"✓ {metric_name}: {actual_value:.4f} >= {target_value}")
                    else:
                        self.logger.warning(f"✗ {metric_name}: {actual_value:.4f} < {target_value}")
                        targets_met = False
                else:
                    self.logger.warning(f"✗ {metric_name}: metric not found in results")
                    targets_met = False
        
        if targets_met:
            self.logger.info("All target metrics achieved!")
        else:
            self.logger.warning("Some target metrics not achieved. Consider additional training.")
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save(path)
        self.logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = SentenceTransformer(path)
        self.logger.info(f"Model loaded from: {path}")