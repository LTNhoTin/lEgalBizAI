"""Training utilities for GPT-OSS 20B finetuning"""

import os
import torch
from datetime import datetime
from typing import Optional, Dict, Any
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer
from config.config import config
from src.data_loader import load_and_prepare_data
import pickle
import hashlib

class GPTTrainer:
    """Main trainer class for GPT-OSS 20B finetuning"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.cache_dir = os.path.join(config.project_root, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup status flags
        self._model_loaded = False
        self._data_loaded = False
        self._trainer_setup = False
    
    def is_model_loaded(self):
        """Check if model is already loaded"""
        return self._model_loaded and self.model is not None and self.tokenizer is not None
    
    def is_data_loaded(self):
        """Check if data is already loaded"""
        return self._data_loaded and self.dataset is not None
    
    def is_trainer_setup(self):
        """Check if trainer is already setup"""
        return self._trainer_setup and self.trainer is not None
    
    def clear_cache(self, cache_type="all"):
        """Clear cached data
        
        Args:
            cache_type (str): Type of cache to clear ('model', 'dataset', 'all')
        """
        if not os.path.exists(self.cache_dir):
            print("🗑️ No cache directory found")
            return
        
        files_removed = 0
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            should_remove = False
            
            if cache_type == "all":
                should_remove = True
            elif cache_type == "model" and filename.startswith("model_"):
                should_remove = True
            elif cache_type == "dataset" and filename.startswith("dataset_"):
                should_remove = True
            
            if should_remove and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    files_removed += 1
                    print(f"🗑️ Removed {filename}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
        
        print(f"✅ Cleared {files_removed} cache files ({cache_type})")
    
    def get_cache_info(self):
        """Get information about cached files"""
        if not os.path.exists(self.cache_dir):
            print("📁 No cache directory found")
            return
        
        model_files = []
        dataset_files = []
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                
                if filename.startswith("model_"):
                    model_files.append((filename, size))
                elif filename.startswith("dataset_"):
                    dataset_files.append((filename, size))
        
        print(f"📁 Cache Directory: {self.cache_dir}")
        print(f"📊 Total Size: {total_size / (1024*1024):.2f} MB")
        print(f"🤖 Model Cache Files: {len(model_files)}")
        for filename, size in model_files:
            print(f"   - {filename} ({size / (1024*1024):.2f} MB)")
        print(f"📋 Dataset Cache Files: {len(dataset_files)}")
        for filename, size in dataset_files:
            print(f"   - {filename} ({size / (1024*1024):.2f} MB)")
        
    def _get_model_cache_key(self):
        """Generate cache key based on model configuration"""
        cache_data = {
            'model_name': config.model.model_name,
            'max_seq_length': config.model.max_seq_length,
            'load_in_4bit': config.model.load_in_4bit,
            'lora_r': config.model.lora_r,
            'lora_alpha': config.model.lora_alpha,
            'lora_dropout': config.model.lora_dropout,
        }
        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _save_model_cache(self):
        """Save model and tokenizer to cache"""
        cache_key = self._get_model_cache_key()
        cache_path = os.path.join(self.cache_dir, f"model_{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model_state': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'config': config.model
                }, f)
            print(f"💾 Model cached to {cache_path}")
        except Exception as e:
            print(f"⚠️ Failed to cache model: {e}")
    
    def _load_model_cache(self):
        """Load model and tokenizer from cache if available"""
        cache_key = self._get_model_cache_key()
        cache_path = os.path.join(self.cache_dir, f"model_{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                print(f"📦 Loading cached model from {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Load base model first
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.model.model_name,
                    dtype=config.model.dtype,
                    max_seq_length=config.model.max_seq_length,
                    load_in_4bit=config.model.load_in_4bit,
                    full_finetuning=config.model.full_finetuning,
                )
                
                # Add LoRA adapters
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=config.model.lora_r,
                    target_modules=config.model.target_modules,
                    lora_alpha=config.model.lora_alpha,
                    lora_dropout=config.model.lora_dropout,
                    bias=config.model.lora_bias,
                    use_gradient_checkpointing=config.model.use_gradient_checkpointing,
                    random_state=config.training.seed,
                    use_rslora=config.model.use_rslora,
                )
                
                print("✅ Model loaded from cache")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load cached model: {e}")
                return False
        return False
    
    def setup_model(self, force_reload=False):
        """Setup model and tokenizer with caching"""
        # Check if model is already loaded and not forcing reload
        if not force_reload and self.is_model_loaded():
            print("✅ Model already loaded, skipping setup...")
            return
            
        print("🚀 Setting up model and tokenizer...")
        print(f"Model: {config.model.model_name}")
        print(f"Max sequence length: {config.model.max_seq_length}")
        print(f"4-bit quantization: {config.model.load_in_4bit}")
        
        # Try to load from cache first
        if not force_reload and self._load_model_cache():
            self._model_loaded = True
            return
        
        print("🔄 Loading model from scratch...")
        # Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.model_name,
            dtype=config.model.dtype,
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.load_in_4bit,
            full_finetuning=config.model.full_finetuning,
        )
        
        # Add LoRA adapters
        print("🔧 Adding LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config.model.lora_r,
            target_modules=config.model.target_modules,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            bias=config.model.lora_bias,
            use_gradient_checkpointing=config.model.use_gradient_checkpointing,
            random_state=config.training.seed,
            use_rslora=config.model.use_rslora,
        )
        
        # Cache the model for future use
        self._save_model_cache()
        self._model_loaded = True
        print("✅ Model setup completed")
        
    def test_model_before_training(self):
        """Test model before training"""
        print("\n🧪 Testing model before training...")
        
        messages = [
            {
                'role': 'system', 
                'content': 'Bạn là NhoTin, lập trình viên tại nhà, luôn đam mê khám phá công nghệ mới, và thích chia sẻ kiến thức với người khác', 
                'thinking': None
            },
            {
                "role": "user", 
                "content": "Bạn thường học công nghệ mới bằng cách nào?"
            },
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        ).to(self.model.device)
        
        print("Response before training:")
        print("-" * 50)
        streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(
            **inputs, 
            max_new_tokens=config.inference.max_new_tokens, 
            streamer=streamer,
            temperature=config.inference.temperature,
            do_sample=config.inference.do_sample
        )
        print("-" * 50)
        
    def _get_data_cache_key(self):
        """Generate cache key for dataset"""
        dataset_path = config.data.dataset_path
        if os.path.exists(dataset_path):
            # Use file modification time and size for cache key
            stat = os.stat(dataset_path)
            cache_data = {
                'dataset_path': dataset_path,
                'mtime': stat.st_mtime,
                'size': stat.st_size,
                'model_name': config.model.model_name
            }
            cache_str = str(sorted(cache_data.items()))
            return hashlib.md5(cache_str.encode()).hexdigest()
        return None
    
    def _save_data_cache(self):
        """Save processed dataset to cache"""
        cache_key = self._get_data_cache_key()
        if cache_key:
            cache_path = os.path.join(self.cache_dir, f"dataset_{cache_key}.pkl")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.dataset, f)
                print(f"💾 Dataset cached to {cache_path}")
            except Exception as e:
                print(f"⚠️ Failed to cache dataset: {e}")
    
    def _load_data_cache(self):
        """Load processed dataset from cache if available"""
        cache_key = self._get_data_cache_key()
        if cache_key:
            cache_path = os.path.join(self.cache_dir, f"dataset_{cache_key}.pkl")
            if os.path.exists(cache_path):
                try:
                    print(f"📦 Loading cached dataset from {cache_path}")
                    with open(cache_path, 'rb') as f:
                        self.dataset = pickle.load(f)
                    print(f"✅ Dataset loaded from cache: {len(self.dataset)} examples")
                    return True
                except Exception as e:
                    print(f"⚠️ Failed to load cached dataset: {e}")
        return False
    
    def load_data(self, force_reload=False):
        """Load and prepare training data with caching support"""
        # Check if data is already loaded and not forcing reload
        if not force_reload and self.is_data_loaded():
            print("✅ Data already loaded, skipping...")
            return
            
        print("\n📊 Loading training data...")
        
        # Try to load from cache first
        if not force_reload and self._load_data_cache():
            self._data_loaded = True
            return
        
        print("🔄 Processing dataset from scratch...")
        self.dataset = load_and_prepare_data(self.tokenizer)
        print(f"✅ Data loaded: {len(self.dataset)} examples")
        
        # Cache the processed dataset
        self._save_data_cache()
        self._data_loaded = True
        
    def setup_trainer(self, force_reload=False):
        """Setup SFT trainer"""
        # Check if trainer is already setup and not forcing reload
        if not force_reload and self.is_trainer_setup():
            print("✅ Trainer already setup, skipping...")
            return
            
        print("\n⚙️ Setting up trainer...")
        
        # Create training arguments - only include max_steps if it's not None
        training_args_dict = {
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "warmup_steps": config.training.warmup_steps,
            "num_train_epochs": config.training.num_train_epochs,
            "learning_rate": config.training.learning_rate,
            "logging_steps": config.training.logging_steps,
            "optim": config.training.optim,
            "weight_decay": config.training.weight_decay,
            "lr_scheduler_type": config.training.lr_scheduler_type,
            "seed": config.training.seed,
            "output_dir": config.training.output_dir,
            "report_to": config.training.report_to,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "dataloader_num_workers": config.training.dataloader_num_workers,
            "remove_unused_columns": config.training.remove_unused_columns,
        }
        
        # Only add max_steps if it's not None
        if config.training.max_steps is not None:
            training_args_dict["max_steps"] = config.training.max_steps
            
        training_args = SFTConfig(**training_args_dict)
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            args=training_args,
        )
        
        print("✅ Trainer setup completed")
        self._trainer_setup = True
        
    def train(self) -> Dict[str, Any]:
        """Start training process"""
        print("\n🏋️ Starting training...")
        print(f"Training epochs: {config.training.num_train_epochs}")
        print(f"Batch size: {config.training.per_device_train_batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Output directory: {config.training.output_dir}")
        
        start_time = datetime.now()
        trainer_stats = self.trainer.train()
        end_time = datetime.now()
        
        training_time = end_time - start_time
        print(f"\n✅ Training completed in {training_time}")
        
        return trainer_stats
        
    def test_model_after_training(self):
        """Test model after training"""
        print("\n🧪 Testing model after training...")
        
        messages = [
            {
                'role': 'system', 
                'content': 'Bạn là Trợ lý Sale Marketing, chuyên phân tích tin nhắn khách hàng và đề xuất chiến lược bán hàng ngắn gọn, thực dụng. Luôn nêu: Ý định, Nỗi lo, Phản hồi gợi ý, CTA, Bước tiếp theo.', 
                'thinking': None
            },
            {
                "role": "user", 
                "content": "Bên em dùng thử CRM 7 ngày rồi nhưng team chưa quen, giá gói Standard là bao nhiêu và có hỗ trợ chuyển dữ liệu không?"
            },
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        ).to(self.model.device)
        
        print("Response after training:")
        print("-" * 50)
        streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(
            **inputs, 
            max_new_tokens=config.inference.max_new_tokens, 
            streamer=streamer,
            temperature=config.inference.temperature,
            do_sample=config.inference.do_sample
        )
        print("-" * 50)
        
    def save_model(self, model_name: str = "finetuned_model"):
        """Save the finetuned model"""
        save_path = os.path.join(config.models_dir, model_name)
        print(f"\n💾 Saving model to {save_path}...")
        
        # Create models directory if it doesn't exist
        os.makedirs(config.models_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✅ Model saved successfully to {save_path}")
        return save_path
        
    def run_full_training_pipeline(self, force_reload=False) -> str:
        """Run the complete training pipeline"""
        print("🚀 Starting GPT-OSS 20B Finetuning Pipeline")
        print("=" * 60)
        
        try:
            # Setup
            self.setup_model(force_reload=force_reload)
            self.test_model_before_training()
            self.load_data(force_reload=force_reload)
            self.setup_trainer(force_reload=force_reload)
            
            # Train
            trainer_stats = self.train()
            
            # Test and save
            self.test_model_after_training()
            model_path = self.save_model()
            
            print("\n🎉 Training pipeline completed successfully!")
            print(f"📁 Model saved at: {model_path}")
            print("=" * 60)
            
            return model_path
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {str(e)}")
            raise e

def main():
    """Main training function"""
    trainer = GPTTrainer()
    model_path = trainer.run_full_training_pipeline()
    return model_path

if __name__ == "__main__":
    main()