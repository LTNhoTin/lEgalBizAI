"""
Data Loader for Embedding Models Fine-tuning
Handles loading and preprocessing QA data for embedding model training
"""

import os
import json
import random
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class EmbeddingDataLoader:
    """Data loader for embedding model fine-tuning with QA pairs"""
    
    def __init__(self, data_path: str, test_split: float = 0.2, seed: int = 42):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the QA dataset JSON file
            test_split: Fraction of data to use for testing (0.0 to 1.0)
            seed: Random seed for reproducible splits
        """
        self.data_path = data_path
        self.test_split = test_split
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Load and split data
        self.raw_data = self._load_data()
        self.train_data, self.test_data = self._split_data()
        
        logger.info(f"Data loaded: {len(self.train_data)} train, {len(self.test_data)} test samples")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load QA data from JSON file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        if not isinstance(data, list):
            raise ValueError("Data should be a list of QA pairs")
        
        # Ensure each item has required fields
        required_fields = ['question', 'answer']
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing field '{field}' in item {i}")
        
        logger.info(f"Loaded {len(data)} QA pairs")
        return data
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and test sets"""
        if self.test_split <= 0:
            return self.raw_data, []
        
        if self.test_split >= 1:
            return [], self.raw_data
        
        train_data, test_data = train_test_split(
            self.raw_data,
            test_size=self.test_split,
            random_state=self.seed,
            shuffle=True
        )
        
        return train_data, test_data
    
    def get_train_data(self) -> List[Dict[str, Any]]:
        """Get training data"""
        return self.train_data
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get test data"""
        return self.test_data
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all data (train + test)"""
        return self.raw_data
    
    def get_evaluation_data(self) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Get data formatted for evaluation
        
        Returns:
            queries: List of questions
            positives: List of corresponding answers
            relevant_docs: List of relevant document indices (for each query)
        """
        test_data = self.test_data if self.test_data else self.train_data
        
        queries = [item['question'] for item in test_data]
        positives = [item['answer'] for item in test_data]
        
        # For QA pairs, each query has one relevant document (its answer)
        relevant_docs = [[i] for i in range(len(queries))]
        
        return queries, positives, relevant_docs
    
    def get_triplets(self, negative_samples: int = 1) -> List[Dict[str, str]]:
        """
        Generate training triplets (query, positive, negative) for contrastive learning
        
        Args:
            negative_samples: Number of negative samples per positive
            
        Returns:
            List of triplets with keys: 'query', 'positive', 'negative'
        """
        triplets = []
        
        for i, item in enumerate(self.train_data):
            query = item['question']
            positive = item['answer']
            
            # Generate negative samples
            for _ in range(negative_samples):
                # Random negative sampling
                neg_idx = random.randint(0, len(self.train_data) - 1)
                while neg_idx == i:  # Ensure negative is different from positive
                    neg_idx = random.randint(0, len(self.train_data) - 1)
                
                negative = self.train_data[neg_idx]['answer']
                
                triplets.append({
                    'query': query,
                    'positive': positive,
                    'negative': negative
                })
        
        return triplets
    
    def create_contrastive_pairs(self, data: List[Dict[str, Any]], num_negatives: int = 1) -> List[Dict[str, Any]]:
        """
        Create contrastive pairs (query, positive, negatives) for training
        
        Args:
            data: List of QA pairs to create contrastive samples from
            num_negatives: Number of negative samples per query
            
        Returns:
            List of contrastive pairs with keys: 'query', 'positive', 'negatives'
        """
        contrastive_pairs = []
        
        for i, item in enumerate(data):
            query = item['question']
            positive = item['answer']
            
            # Generate negative samples
            negatives = []
            for _ in range(num_negatives):
                # Random negative sampling from other answers
                neg_idx = random.randint(0, len(data) - 1)
                while neg_idx == i:  # Ensure negative is different from current item
                    neg_idx = random.randint(0, len(data) - 1)
                
                negative = data[neg_idx]['answer']
                negatives.append(negative)
            
            contrastive_pairs.append({
                'query': query,
                'positive': positive,
                'negatives': negatives
            })
        
        return contrastive_pairs
    
    def get_sentence_transformer_examples(self, negative_samples: int = 1):
        """
        Get data in SentenceTransformer InputExample format
        
        Args:
            negative_samples: Number of negative samples per positive
            
        Returns:
            List of InputExample objects
        """
        try:
            from sentence_transformers import InputExample
        except ImportError:
            raise ImportError("sentence-transformers is required for this method")
        
        examples = []
        triplets = self.get_triplets(negative_samples)
        
        for triplet in triplets:
            examples.append(InputExample(
                texts=[triplet['query'], triplet['positive']],
                label=1.0  # Positive pair
            ))
            
            examples.append(InputExample(
                texts=[triplet['query'], triplet['negative']],
                label=0.0  # Negative pair
            ))
        
        return examples
    
    def print_data_info(self):
        """Print detailed information about the loaded data"""
        print("=" * 60)
        print("DATA LOADER INFORMATION")
        print("=" * 60)
        print(f"Data path: {self.data_path}")
        print(f"Test split: {self.test_split}")
        print(f"Random seed: {self.seed}")
        print(f"Total samples: {len(self.raw_data)}")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        if self.train_data:
            # Analyze question types if available
            question_types = {}
            for item in self.train_data:
                q_type = item.get('type', 'unknown')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            print("\nQuestion types in training data:")
            for q_type, count in question_types.items():
                percentage = (count / len(self.train_data)) * 100
                print(f"  {q_type}: {count} ({percentage:.1f}%)")
            
            # Show sample data
            print("\nSample training data:")
            sample = self.train_data[0]
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Answer: {sample['answer'][:100]}...")
        
        print("=" * 60)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the data"""
        stats = {
            'total_samples': len(self.raw_data),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'test_split_ratio': self.test_split,
        }
        
        if self.train_data:
            # Question length statistics
            question_lengths = [len(item['question'].split()) for item in self.train_data]
            answer_lengths = [len(item['answer'].split()) for item in self.train_data]
            
            stats.update({
                'avg_question_length': np.mean(question_lengths),
                'avg_answer_length': np.mean(answer_lengths),
                'max_question_length': max(question_lengths),
                'max_answer_length': max(answer_lengths),
                'min_question_length': min(question_lengths),
                'min_answer_length': min(answer_lengths),
            })
            
            # Question types if available
            question_types = {}
            for item in self.train_data:
                q_type = item.get('type', 'unknown')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            stats['question_types'] = question_types
        
        return stats
    
    def save_split_data(self, output_dir: str):
        """Save train/test split to separate files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train data
        train_path = output_dir / "train_data.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, ensure_ascii=False, indent=2)
        
        # Save test data
        test_path = output_dir / "test_data.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data splits saved to {output_dir}")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Test: {test_path}")


# Alias for backward compatibility
QADataLoader = EmbeddingDataLoader


if __name__ == "__main__":
    # Example usage
    data_path = "/home/nhotin/work/LegalBizAI_project/finetune/data_finetunning/qaset_full_article.json"
    
    if os.path.exists(data_path):
        loader = EmbeddingDataLoader(data_path, test_split=0.2)
        loader.print_data_info()
        
        # Test triplet generation
        triplets = loader.get_triplets(negative_samples=2)
        print(f"\nGenerated {len(triplets)} training triplets")
        
        # Test evaluation data
        queries, positives, relevant_docs = loader.get_evaluation_data()
        print(f"Evaluation data: {len(queries)} queries, {len(positives)} positives")
        
        # Show statistics
        stats = loader.get_data_statistics()
        print(f"\nData statistics: {stats}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable to point to your QA dataset")