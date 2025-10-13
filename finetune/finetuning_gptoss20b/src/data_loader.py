"""Data loading and preprocessing utilities"""

import os
import json
from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
from config.config import config

class DataLoader:
    """Handle data loading and preprocessing for finetuning"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or config.data.dataset_path
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for data processing"""
        self.tokenizer = tokenizer
        
    def load_dataset(self) -> Dataset:
        """Load dataset from JSONL file"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
        print(f"Loading dataset from {self.dataset_path}")
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        print(f"Loaded {len(dataset)} examples")
        
        return dataset
    
    def standardize_dataset(self, dataset: Dataset) -> Dataset:
        """Standardize dataset format using ShareGPT format"""
        print("Standardizing dataset format...")
        dataset = standardize_sharegpt(dataset)
        return dataset
    
    def formatting_prompts_func(self, examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """Format prompts for training"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
            
        convos = examples["messages"]
        texts = [
            self.tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ) for convo in convos
        ]
        return {"text": texts}
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for training"""
        print("Processing dataset...")
        
        # Standardize format
        dataset = self.standardize_dataset(dataset)
        
        # Apply formatting
        dataset = dataset.map(
            self.formatting_prompts_func, 
            batched=True,
            desc="Formatting prompts"
        )
        
        print(f"Processed {len(dataset)} examples")
        print("Sample formatted text:")
        print(dataset[0]['text'][:500] + "..." if len(dataset[0]['text']) > 500 else dataset[0]['text'])
        
        return dataset
    
    def get_training_dataset(self) -> Dataset:
        """Get processed dataset ready for training"""
        dataset = self.load_dataset()
        dataset = self.process_dataset(dataset)
        return dataset
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """Validate dataset format and content"""
        print("Validating dataset...")
        
        # Check if dataset is not empty
        if len(dataset) == 0:
            print("Dataset is empty")
            return False
            
        # Check required columns
        required_columns = ['text']
        for col in required_columns:
            if col not in dataset.column_names:
                print(f"Missing required column: {col}")
                return False
                
        # Check sample data
        sample = dataset[0]
        if not sample['text'] or len(sample['text'].strip()) == 0:
            print("Empty text in sample data")
            return False
            
        print(f"Dataset validation passed. {len(dataset)} examples ready for training.")
        return True

def load_and_prepare_data(tokenizer) -> Dataset:
    """Convenience function to load and prepare data"""
    data_loader = DataLoader()
    data_loader.set_tokenizer(tokenizer)
    dataset = data_loader.get_training_dataset()
    
    if not data_loader.validate_dataset(dataset):
        raise ValueError("Dataset validation failed")
        
    return dataset

if __name__ == "__main__":
    # Test data loading
    data_loader = DataLoader()
    dataset = data_loader.load_dataset()
    print(f"Dataset loaded successfully with {len(dataset)} examples")
    print("Sample data:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))