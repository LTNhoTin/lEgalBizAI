"""GPT-OSS 20B Finetuning Package"""

__version__ = "1.0.0"
__author__ = "NhoTin"
__description__ = "GPT-OSS 20B Finetuning for Sale Marketing"

from .data_loader import DataLoader, load_and_prepare_data
from .trainer import GPTTrainer

__all__ = [
    "DataLoader",
    "load_and_prepare_data", 
    "GPTTrainer"
]