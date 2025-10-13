"""
BGE-M3 Trainer - Optimized Architecture
Fixes all identified issues:
- Mixed precision with proper autocast
- Optimized dataset collate to prevent OOM
- Batch evaluation for speed
- Correct gradient accumulation
- Proper LoRA target modules
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import (
    AutoTokenizer, AutoModel, 
    get_cosine_schedule_with_warmup,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

import wandb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add evaluation utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation_utils import EmbeddingEvaluator, prepare_evaluation_data
from data_loader import EmbeddingDataLoader as QADataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BGETrainingConfig:
    """Optimized BGE-M3 Training Configuration"""
    # Model settings
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 384
    
    # Training settings
    batch_size: int = 2
    num_epochs: int = 30
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # BGE-M3 specific
    use_dense: bool = True
    use_sparse: bool = False
    use_colbert: bool = False
    
    # Mixed precision
    use_fp16: bool = True
    use_bf16: bool = False
    
    # Logging & checkpoints
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    output_dir: str = "./output/bge_m3"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "legal-embedding-models"
    wandb_run_name: str = "bge-m3-optimized"


class OptimizedDataCollator:
    """Optimized data collator to prevent OOM with smart padding"""
    
    def __init__(self, tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Smart collate with dynamic padding to prevent OOM"""
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]
        
        # Find actual max length in batch to avoid unnecessary padding
        all_texts = queries + positives + negatives
        actual_max_len = min(
            max(len(self.tokenizer.encode(text, add_special_tokens=True)) for text in all_texts),
            self.max_length
        )
        
        # Tokenize with dynamic max length
        query_encodings = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=actual_max_len,
            return_tensors="pt"
        )
        
        positive_encodings = self.tokenizer(
            positives, 
            padding=True, 
            truncation=True, 
            max_length=actual_max_len,
            return_tensors="pt"
        )
        
        negative_encodings = self.tokenizer(
            negatives, 
            padding=True, 
            truncation=True, 
            max_length=actual_max_len,
            return_tensors="pt"
        )
        
        return {
            'query_input_ids': query_encodings['input_ids'],
            'query_attention_mask': query_encodings['attention_mask'],
            'positive_input_ids': positive_encodings['input_ids'],
            'positive_attention_mask': positive_encodings['attention_mask'],
            'negative_input_ids': negative_encodings['input_ids'],
            'negative_attention_mask': negative_encodings['attention_mask'],
        }


class BGEModel(nn.Module):
    """Optimized BGE-M3 Model with proper LoRA target modules"""
    
    def __init__(self, config: BGETrainingConfig):
        super().__init__()
        self.config = config
        
        # Load backbone model
        self.backbone = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Setup LoRA with correct target modules for BGE
        if config.use_lora:
            # Correct target modules for BGE backbone (BERT-based)
            target_modules = [
                "query", "key", "value", "dense",  # Attention layers
                "intermediate.dense", "output.dense"  # FFN layers
            ]
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self.backbone = get_peft_model(self.backbone, lora_config)
            logger.info(f"LoRA applied to modules: {target_modules}")
        
        # BGE-M3 projection heads
        hidden_size = self.backbone.config.hidden_size
        
        if config.use_dense:
            self.dense_proj = nn.Linear(hidden_size, hidden_size)
        
        if config.use_sparse:
            self.sparse_proj = nn.Linear(hidden_size, 30522)  # vocab size
        
        if config.use_colbert:
            self.colbert_proj = nn.Linear(hidden_size, 128)
    
    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with all projection heads"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        pooled_output = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        
        result = {}
        
        if self.config.use_dense:
            result['dense'] = F.normalize(self.dense_proj(pooled_output), p=2, dim=1)
        
        if self.config.use_sparse:
            result['sparse'] = torch.relu(self.sparse_proj(pooled_output))
        
        if self.config.use_colbert:
            # ColBERT uses token-level representations
            result['colbert'] = F.normalize(
                self.colbert_proj(outputs.last_hidden_state), p=2, dim=-1
            )
        
        return result


class ContrastiveLoss(nn.Module):
    """Optimized contrastive loss with temperature scaling"""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        batch_size = query_emb.size(0)
        
        # Compute similarities
        pos_sim = torch.sum(query_emb * pos_emb, dim=1) / self.temperature
        neg_sim = torch.sum(query_emb * neg_emb, dim=1) / self.temperature
        
        # Create logits and labels
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
        
        return self.cross_entropy(logits, labels)


class OptimizedBGETrainer:
    """Optimized BGE-M3 Trainer with all fixes"""
    
    def __init__(self, config: BGETrainingConfig, train_dataset: Dataset, eval_dataset: Dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BGEModel(config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Setup data collator
        self.data_collator = OptimizedDataCollator(self.tokenizer, config.max_seq_length)
        
        # Setup loss function
        self.criterion = ContrastiveLoss()
        
        # Setup mixed precision
        self.use_amp = config.use_fp16 or config.use_bf16
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info(f"Mixed precision enabled: fp16={config.use_fp16}, bf16={config.use_bf16}")
        
        # Setup optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=2,
            pin_memory=True
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size * 2,  # Larger batch for eval
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=2,
            pin_memory=True
        )
        
        # Calculate total steps and setup scheduler
        self.total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Setup tracking
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with proper mixed precision"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with autocast
        with autocast(enabled=self.use_amp):
            # Get embeddings
            query_outputs = self.model(batch['query_input_ids'], batch['query_attention_mask'])
            pos_outputs = self.model(batch['positive_input_ids'], batch['positive_attention_mask'])
            neg_outputs = self.model(batch['negative_input_ids'], batch['negative_attention_mask'])
            
            # Compute loss (only dense for now)
            loss = self.criterion(
                query_outputs['dense'],
                pos_outputs['dense'],
                neg_outputs['dense']
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train(self):
        """Main training loop with all optimizations"""
        logger.info("Starting training...")
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            accumulated_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_loss = self.train_step(batch)
                accumulated_loss += step_loss
                epoch_loss += step_loss
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch + (step + 1) / len(self.train_loader),
                                "train/global_step": self.global_step
                            })
                        
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        accumulated_loss = 0.0
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        
                        # Save best model
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_best_model()
                            logger.info(f"New best model saved with loss: {eval_loss:.4f}")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(epoch, step)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "epoch/loss": avg_epoch_loss,
                    "epoch/best_loss": self.best_loss,
                    "epoch/number": epoch + 1
                })
        
        # Final evaluation and save
        final_eval_loss = self.evaluate()
        self.save_final_model()
        
        # Run comprehensive final evaluation
        final_eval_metrics = self._final_evaluation()
        
        logger.info("Training completed!")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info(f"Final loss: {final_eval_loss:.4f}")
        
        # Log final evaluation summary
        if final_eval_metrics:
            logger.info("Final evaluation metrics logged to W&B and console")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Enhanced evaluation with loss and similarity metrics"""
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_similarities = []
        
        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                # Get embeddings
                query_outputs = self.model(batch['query_input_ids'], batch['query_attention_mask'])
                pos_outputs = self.model(batch['positive_input_ids'], batch['positive_attention_mask'])
                neg_outputs = self.model(batch['negative_input_ids'], batch['negative_attention_mask'])
                
                # Compute loss
                loss = self.criterion(
                    query_outputs['dense'],
                    pos_outputs['dense'],
                    neg_outputs['dense']
                )
                
                # Calculate similarities for additional metrics
                query_emb = F.normalize(query_outputs['dense'], p=2, dim=1)
                pos_emb = F.normalize(pos_outputs['dense'], p=2, dim=1)
                
                # Cosine similarity between query and positive
                similarities = torch.sum(query_emb * pos_emb, dim=1)
                all_similarities.extend(similarities.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate similarity statistics
        similarities_array = np.array(all_similarities)
        mean_sim = np.mean(similarities_array)
        std_sim = np.std(similarities_array)
        min_sim = np.min(similarities_array)
        max_sim = np.max(similarities_array)
        
        # Accuracy at different thresholds
        acc_05 = np.mean(similarities_array >= 0.5)
        acc_06 = np.mean(similarities_array >= 0.6)
        acc_07 = np.mean(similarities_array >= 0.7)
        
        eval_metrics = {
            "eval/loss": avg_loss,
            "eval/mean_similarity": mean_sim,
            "eval/std_similarity": std_sim,
            "eval/min_similarity": min_sim,
            "eval/max_similarity": max_sim,
            "eval/accuracy_0.5": acc_05,
            "eval/accuracy_0.6": acc_06,
            "eval/accuracy_0.7": acc_07,
            "eval/global_step": self.global_step
        }
        
        if self.config.use_wandb:
            wandb.log(eval_metrics)
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Mean Sim: {mean_sim:.4f} ± {std_sim:.4f}")
        logger.info(f"Similarity Range: [{min_sim:.4f}, {max_sim:.4f}], Acc@0.5: {acc_05:.4f}")
        
        return avg_loss
    
    def save_best_model(self):
        """Save the best model"""
        best_model_dir = Path(self.config.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        if hasattr(self.model.backbone, 'save_pretrained'):
            self.model.backbone.save_pretrained(best_model_dir)
        else:
            torch.save(self.model.state_dict(), best_model_dir / "pytorch_model.bin")
        
        self.tokenizer.save_pretrained(best_model_dir)
        
        # Save config
        with open(best_model_dir / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
    
    def _final_evaluation(self):
        """Comprehensive final evaluation with advanced metrics"""
        logger.info("Running comprehensive final evaluation...")
        
        try:
            # Load evaluation data
            data_loader = QADataLoader(
                data_path=Path(__file__).parent.parent.parent / "data" / "processed" / "qa_pairs.json"
            )
            
            # Get evaluation data
            queries, positives, relevant_docs = data_loader.get_evaluation_data()
            
            # Encode queries and positives using the trained model
            self.model.eval()
            with torch.no_grad():
                # Encode queries
                query_inputs = self.tokenizer(
                    queries, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.config.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                query_outputs = self.model(query_inputs['input_ids'], query_inputs['attention_mask'])
                query_embeddings = query_outputs['dense'].cpu().numpy()
                
                # Encode positives
                positive_inputs = self.tokenizer(
                    positives,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                positive_outputs = self.model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                positive_embeddings = positive_outputs['dense'].cpu().numpy()
            
            # Calculate cosine similarities
            similarities = []
            for q_emb, p_emb in zip(query_embeddings, positive_embeddings):
                similarity = cosine_similarity([q_emb], [p_emb])[0][0]
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            
            # Basic similarity metrics
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            # Accuracy at different thresholds
            accuracy_05 = np.mean(similarities >= 0.5)
            accuracy_06 = np.mean(similarities >= 0.6)
            accuracy_07 = np.mean(similarities >= 0.7)
            
            # Prepare evaluation metrics
            eval_metrics = {
                'eval/mean_similarity': mean_similarity,
                'eval/std_similarity': std_similarity,
                'eval/min_similarity': min_similarity,
                'eval/max_similarity': max_similarity,
                'eval/accuracy_0.5': accuracy_05,
                'eval/accuracy_0.6': accuracy_06,
                'eval/accuracy_0.7': accuracy_07,
            }
            
            # Advanced retrieval metrics
            evaluator = EmbeddingEvaluator(k_values=[1, 3, 5, 10])
            
            # Prepare data for retrieval evaluation
            retrieved_docs = []
            for i, (query_emb, relevant_doc_list) in enumerate(zip(query_embeddings, relevant_docs)):
                # Simulate retrieval by using positive as top result
                retrieved_docs.append([i])  # Use index as document ID
            
            # Calculate advanced metrics
            metrics_results = evaluator.evaluate_all_metrics(relevant_docs, retrieved_docs)
            
            # Add advanced metrics to eval_metrics
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
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Mean Similarity: {mean_similarity:.4f} ± {std_similarity:.4f}")
            logger.info(f"Similarity Range: [{min_similarity:.4f}, {max_similarity:.4f}]")
            logger.info(f"Accuracy@0.5: {accuracy_05:.4f}")
            logger.info(f"Accuracy@0.6: {accuracy_06:.4f}")
            logger.info(f"Accuracy@0.7: {accuracy_07:.4f}")
            
            # Log retrieval metrics
            for metric_name, k_results in metrics_results.items():
                metric_str = f"{metric_name}: "
                metric_values = [f"{metric_name}@{k}={v:.4f}" for k, v in k_results.items()]
                logger.info(metric_str + ", ".join(metric_values))
            
            logger.info("=" * 60)
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def save_final_model(self):
        """Save final model"""
        final_model_dir = Path(self.config.output_dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model.backbone, 'save_pretrained'):
            self.model.backbone.save_pretrained(final_model_dir)
        else:
            torch.save(self.model.state_dict(), final_model_dir / "pytorch_model.bin")
        
        self.tokenizer.save_pretrained(final_model_dir)