"""
Data loader cho GPT-OSS 20B fine-tuning
Load và xử lý data từ qaset_full_article.json
"""
import json
import random
from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalDataLoader:
    """Load và xử lý Legal QA data"""
    
    def __init__(
        self,
        data_path: str,
        max_samples: int = 2000,
        train_split: float = 0.8,
        random_seed: int = 42
    ):
        """
        Args:
            data_path: Đường dẫn tới file JSON
            max_samples: Số lượng samples tối đa (default 2000)
            train_split: Tỉ lệ train (default 0.8 = 80%)
            random_seed: Random seed
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.train_split = train_split
        self.random_seed = random_seed
        
        random.seed(random_seed)
        
        logger.info(f"Loading data from: {data_path}")
        logger.info(f"Max samples: {max_samples}")
        logger.info(f"Train/Test split: {train_split}/{1-train_split}")
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load data và chia train/test
        
        Returns:
            train_data, test_data
        """
        # Load JSON
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} total samples")
        
        # Sample random nếu data lớn hơn max_samples
        if len(data) > self.max_samples:
            data = random.sample(data, self.max_samples)
            logger.info(f"Sampled {self.max_samples} samples")
        
        # Shuffle data
        random.shuffle(data)
        
        # Split train/test
        split_idx = int(len(data) * self.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        
        return train_data, test_data
    
    def format_prompt(self, sample: Dict, prompt_template: str) -> str:
        """
        Format sample thành prompt
        
        Args:
            sample: Dictionary chứa question và answer
            prompt_template: Template string
            
        Returns:
            Formatted prompt string
        """
        return prompt_template.format(
            question=sample['question'],
            answer=sample['answer']
        )
    
    def create_datasets(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_template: str,
        max_length: int = 2048
    ) -> Tuple[Dataset, Dataset]:
        """
        Tạo HuggingFace Dataset từ data
        
        Args:
            tokenizer: Tokenizer
            prompt_template: Template cho prompt
            max_length: Max sequence length
            
        Returns:
            train_dataset, test_dataset
        """
        train_data, test_data = self.load_data()
        
        # Format prompts
        train_texts = [
            self.format_prompt(sample, prompt_template) 
            for sample in train_data
        ]
        test_texts = [
            self.format_prompt(sample, prompt_template) 
            for sample in test_data
        ]
        
        # Tokenize
        logger.info("Tokenizing train data...")
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        logger.info("Tokenizing test data...")
        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_encodings['input_ids']  # For causal LM
        })
        
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_encodings['input_ids']
        })
        
        logger.info(f"Created train dataset with {len(train_dataset)} samples")
        logger.info(f"Created test dataset with {len(test_dataset)} samples")
        
        return train_dataset, test_dataset
    
    def print_sample(self, tokenizer: PreTrainedTokenizer, prompt_template: str):
        """In ra sample để kiểm tra"""
        train_data, test_data = self.load_data()
        
        if train_data:
            sample = train_data[0]
            formatted = self.format_prompt(sample, prompt_template)
            
            logger.info("\n" + "="*80)
            logger.info("SAMPLE DATA:")
            logger.info("="*80)
            logger.info(f"\nOriginal question: {sample['question'][:100]}...")
            logger.info(f"\nOriginal answer: {sample['answer'][:200]}...")
            logger.info("\n" + "-"*80)
            logger.info("FORMATTED PROMPT:")
            logger.info("-"*80)
            logger.info(formatted[:500] + "..." if len(formatted) > 500 else formatted)
            logger.info("\n" + "="*80)
            
            # Tokenize sample
            tokens = tokenizer(formatted, truncation=True, max_length=2048)
            logger.info(f"\nTokenized length: {len(tokens['input_ids'])} tokens")
            logger.info("="*80 + "\n")

