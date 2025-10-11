"""Configuration file for GPT-OSS 20B finetuning project"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "unsloth/gpt-oss-20b"
    max_seq_length: int = 1024
    dtype: Optional[str] = None  # None for auto detection
    load_in_4bit: bool = True
    full_finetuning: bool = False
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules: List[str] = None
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration"""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = "data/sale_marketing_finetune_dataset.jsonl"
    test_size: float = 0.1
    random_state: int = 42
    
@dataclass
class InferenceConfig:
    """Inference configuration"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    
@dataclass
class ProjectConfig:
    """Main project configuration"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    inference: InferenceConfig = None
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
            
        # Update paths to be absolute
        self.models_dir = os.path.join(self.project_root, self.models_dir)
        self.logs_dir = os.path.join(self.project_root, self.logs_dir)
        self.training.output_dir = os.path.join(self.project_root, self.training.output_dir)
        self.data.dataset_path = os.path.join(self.project_root, self.data.dataset_path)

# Default configuration instance
config = ProjectConfig()

# System messages for different personas
SYSTEM_MESSAGES = {
    "sale_marketing": "Bạn là Trợ lý Sale Marketing, chuyên phân tích tin nhắn khách hàng và đề xuất chiến lược bán hàng ngắn gọn, thực dụng. Luôn nêu: Ý định, Nỗi lo, Phản hồi gợi ý, CTA, Bước tiếp theo.",
    "nhotin": "Bạn là NhoTin, lập trình viên tại nhà, luôn đam mê khám phá công nghệ mới, và thích chia sẻ kiến thức với người khác",
    "thang_nguyen": "Bạn là Thang Nguyen, lập trình viên tại Ngân hàng SHB, luôn đam mê khám phá công nghệ mới"
}