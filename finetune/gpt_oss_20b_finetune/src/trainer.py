"""
Trainer cho GPT-OSS 20B với LoRA/QLoRA
"""
import os
import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTOSSTrainer:
    """Trainer class cho GPT-OSS 20B fine-tuning với LoRA/QLoRA"""
    
    def __init__(self, config):
        """
        Args:
            config: Config object chứa tất cả cấu hình
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        """Setup model và tokenizer với quantization và LoRA"""
        logger.info("Setting up model and tokenizer...")
        
        # BitsAndBytes config cho 4-bit quantization
        bnb_config = None
        if self.config.model.load_in_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.model.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
            )
        elif self.config.model.load_in_8bit:
            logger.info("Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Set pad token nếu chưa có
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token = eos_token")
        
        # Load model
        logger.info(f"Loading model: {self.config.model.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float16
        )
        
        # Prepare model cho k-bit training
        if bnb_config is not None:
            logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete!")
        
        return self.model, self.tokenizer
    
    def print_trainable_parameters(self):
        """In ra số lượng trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def create_trainer(self, train_dataset, eval_dataset):
        """
        Tạo Hugging Face Trainer
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        logger.info("Creating trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            optim=self.config.training.optim,
            max_grad_norm=self.config.training.max_grad_norm,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            eval_strategy=self.config.training.evaluation_strategy,  # Changed from evaluation_strategy (deprecated in transformers 4.56+)
            save_strategy=self.config.training.save_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            seed=self.config.training.seed,
            report_to=self.config.training.report_to,
            logging_dir=os.path.join(self.config.project_root, "logs"),
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer created!")
        
        return self.trainer
    
    def train(self):
        """Bắt đầu training"""
        if self.trainer is None:
            raise ValueError("Trainer chưa được tạo. Gọi create_trainer() trước.")
        
        logger.info("Starting training...")
        logger.info("="*80)
        
        # Train
        self.trainer.train()
        
        logger.info("="*80)
        logger.info("Training completed!")
        
        return self.trainer
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Lưu model
        
        Args:
            output_dir: Thư mục output, nếu None sẽ dùng từ config
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.training.output_dir,
                "final_model"
            )
        
        logger.info(f"Saving model to {output_dir}...")
        
        # Save model và tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved!")
        
        return output_dir
    
    def evaluate(self):
        """Evaluate model"""
        if self.trainer is None:
            raise ValueError("Trainer chưa được tạo.")
        
        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate()
        
        logger.info("Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics

