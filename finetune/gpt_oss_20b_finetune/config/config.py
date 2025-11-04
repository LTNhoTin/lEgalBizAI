"""
Cấu hình training cho GPT-OSS 20B với LoRA/QLoRA
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Cấu hình model"""
    model_name: str = "pansophic/rocket-3B"  # GPT-OSS 20B model
    load_in_4bit: bool = True  # Sử dụng 4-bit quantization để fit vào 16GB VRAM
    load_in_8bit: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    # BitsAndBytes config
    bnb_4bit_compute_dtype: str = "float16"  # bfloat16 hoặc float16
    bnb_4bit_quant_type: str = "nf4"  # nf4 hoặc fp4
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    
@dataclass
class LoRAConfig:
    """Cấu hình LoRA"""
    r: int = 64  # Rank của LoRA
    lora_alpha: int = 16  # Alpha parameter
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Test config với rank nhỏ hơn
    test_r: int = 8
    test_lora_alpha: int = 16
    
@dataclass
class DataConfig:
    """Cấu hình data"""
    data_path: str = "../data_finetunning/qaset_full_article.json"
    max_samples: int = 2000  # Giảm xuống 2k samples
    train_split: float = 0.8  # 80% train
    test_split: float = 0.2   # 20% test
    random_seed: int = 42
    
    # Text processing
    max_length: int = 2048
    prompt_template: str = """### Câu hỏi:
{question}

### Trả lời:
{answer}"""
    
    # Test config với samples nhỏ hơn
    test_max_samples: int = 100
    test_max_length: int = 512
    
@dataclass
class TrainingConfig:
    """Cấu hình training"""
    output_dir: str = "./outputs"
    
    # Training hyperparameters
    num_train_epochs: int = 10  # Giảm từ 20 → 10 epochs để tránh overfit
    per_device_train_batch_size: int = 4  # Tăng từ 1 lên 4 để tận dụng VRAM
    per_device_eval_batch_size: int = 4   # Tăng eval batch size
    gradient_accumulation_steps: int = 8  # Tăng từ 4 → 8 (effective batch = 32)
    
    # Optimization
    learning_rate: float = 1.2e-4  # Giảm từ 2e-4 → 1.2e-4 để ổn định hơn
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"  # Optimizer tối ưu cho memory
    
    # Gradient & memory
    max_grad_norm: float = 1.0  # Tăng từ 0.3 → 1.0 để clip gradient tốt hơn
    gradient_checkpointing: bool = False  # Tắt để train nhanh hơn (dùng ~8-10GB VRAM)
    fp16: bool = False  # Tắt fp16
    bf16: bool = True  # Bật bf16 để giảm NaN (GPU hỗ trợ)
    
    # Logging & Evaluation
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Other
    seed: int = 42
    report_to: str = "wandb"
    
    # Test config
    test_num_train_epochs: int = 1
    test_per_device_train_batch_size: int = 1
    test_gradient_accumulation_steps: int = 4
    test_logging_steps: int = 5
    test_eval_steps: int = 20
    test_save_steps: int = 20  # Must be multiple of eval_steps when load_best_model_at_end=True
    
@dataclass
class WandbConfig:
    """Cấu hình Wandb"""
    project: str = "gpt-oss-20b-legal-finetune"
    entity: Optional[str] = None  # Set your wandb username/team
    name: Optional[str] = None  # Run name, will be auto-generated
    tags: List[str] = field(default_factory=lambda: ["gpt-oss", "legal", "vietnamese"])
    notes: str = "Fine-tuning GPT-OSS 20B on Vietnamese Legal QA dataset"
    
@dataclass
class Config:
    """Main config class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def __post_init__(self):
        # Update paths to absolute
        self.data.data_path = os.path.join(self.project_root, self.data.data_path)
        self.training.output_dir = os.path.join(self.project_root, self.training.output_dir)
        
        # Create directories
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "data"), exist_ok=True)
    
    def get_test_config(self):
        """Trả về config cho test mode"""
        test_config = Config()
        
        # Update LoRA config cho test
        test_config.lora.r = self.lora.test_r
        test_config.lora.lora_alpha = self.lora.test_lora_alpha
        
        # Update data config cho test
        test_config.data.max_samples = self.data.test_max_samples
        test_config.data.max_length = self.data.test_max_length
        
        # Update training config cho test
        test_config.training.num_train_epochs = self.training.test_num_train_epochs
        test_config.training.per_device_train_batch_size = self.training.test_per_device_train_batch_size
        test_config.training.gradient_accumulation_steps = self.training.test_gradient_accumulation_steps
        test_config.training.logging_steps = self.training.test_logging_steps
        test_config.training.eval_steps = self.training.test_eval_steps
        test_config.training.save_steps = self.training.test_save_steps
        test_config.training.output_dir = os.path.join(self.project_root, "outputs_test")
        
        # Update wandb name
        test_config.wandb.name = "test_run"
        test_config.wandb.tags = self.wandb.tags + ["test"]
        
        return test_config

# Global config instance
config = Config()

