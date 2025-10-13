"""
Data loader for embedding model finetuning
Processes QA dataset for contrastive learning
"""

import json
import random
from typing import List, Dict, Tuple, Any
import logging
import yaml
import os
from pathlib import Path
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

class EmbeddingDataLoader:
    """Data loader for embedding model training"""
    
    def __init__(self, data_path: str, test_split: float = 0.2, random_seed: int = 42):
        """
        Initialize data loader
        
        Args:
            data_path: Path to QA dataset JSON file
            test_split: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_split = test_split
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.qa_data = self._load_qa_data()
        self.train_data, self.test_data = self._split_data()
        
    def _load_qa_data(self) -> List[Dict]:
        """Load QA data from JSON file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} QA pairs from {self.data_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and test sets"""
        data_copy = self.qa_data.copy()
        random.shuffle(data_copy)
        
        split_idx = int(len(data_copy) * (1 - self.test_split))
        train_data = data_copy[:split_idx]
        test_data = data_copy[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
    
    def create_contrastive_pairs(self, data: List[Dict], num_negatives: int = 3) -> List[Dict]:
        """
        Create contrastive learning pairs (query, positive, negatives)
        
        Args:
            data: List of QA dictionaries
            num_negatives: Number of negative examples per positive pair
            
        Returns:
            List of contrastive training examples
        """
        contrastive_pairs = []
        
        # Create a mapping of chunk_id to answer for negative sampling
        chunk_to_answer = {}
        for item in data:
            for chunk_id in item.get('chunk_ids', []):
                chunk_to_answer[chunk_id] = item['answer']
        
        all_answers = list(chunk_to_answer.values())
        
        for item in data:
            query = item['question']
            positive_answer = item['answer']
            
            # Sample negative answers (different from positive)
            negative_candidates = [ans for ans in all_answers if ans != positive_answer]
            negatives = random.sample(negative_candidates, min(num_negatives, len(negative_candidates)))
            
            contrastive_pairs.append({
                'query': query,
                'positive': positive_answer,
                'negatives': negatives,
                'chunk_ids': item.get('chunk_ids', []),
                'type_question': item.get('type_question', 'query')
            })
        
        logger.info(f"Created {len(contrastive_pairs)} contrastive pairs")
        return contrastive_pairs
    
    def create_sentence_transformer_dataset(self, data: List[Dict]) -> Dataset:
        """
        Create dataset for sentence-transformers training
        
        Args:
            data: List of QA dictionaries
            
        Returns:
            HuggingFace Dataset object
        """
        contrastive_pairs = self.create_contrastive_pairs(data)
        
        # Format for sentence-transformers InputExample
        examples = []
        for pair in contrastive_pairs:
            # Positive pair
            examples.append({
                'sentence1': pair['query'],
                'sentence2': pair['positive'],
                'label': 1.0
            })
            
            # Negative pairs
            for negative in pair['negatives']:
                examples.append({
                    'sentence1': pair['query'],
                    'sentence2': negative,
                    'label': 0.0
                })
        
        return Dataset.from_list(examples)
    
    def create_triplet_dataset(self, data: List[Dict]) -> List[Tuple[str, str, str]]:
        """
        Create triplet dataset (anchor, positive, negative)
        
        Args:
            data: List of QA dictionaries
            
        Returns:
            List of triplets (anchor, positive, negative)
        """
        contrastive_pairs = self.create_contrastive_pairs(data)
        
        triplets = []
        for pair in contrastive_pairs:
            anchor = pair['query']
            positive = pair['positive']
            
            for negative in pair['negatives']:
                triplets.append((anchor, positive, negative))
        
        logger.info(f"Created {len(triplets)} triplets")
        return triplets
    
    def get_evaluation_data(self) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Get evaluation data (queries, documents, relevant_docs)
        
        Returns:
            Tuple of (queries, documents, relevant_doc_indices)
        """
        queries = []
        documents = []
        relevant_docs = []
        
        # Create document mapping
        doc_to_idx = {}
        idx = 0
        
        for item in self.test_data:
            query = item['question']
            answer = item['answer']
            chunk_ids = item.get('chunk_ids', [])
            
            queries.append(query)
            
            # Add answer as document if not already present
            if answer not in doc_to_idx:
                doc_to_idx[answer] = idx
                documents.append(answer)
                idx += 1
            
            # Map chunk_ids to document indices
            relevant_doc_indices = [doc_to_idx[answer]]  # For now, use answer as relevant doc
            relevant_docs.append(relevant_doc_indices)
        
        return queries, documents, relevant_docs
    
    def save_processed_data(self, output_dir: str) -> None:
        """
        Save processed data to files
        
        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save train/test split
        with open(output_path / 'train_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, ensure_ascii=False, indent=2)
            
        with open(output_path / 'test_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=2)
        
        # Save contrastive pairs
        train_contrastive = self.create_contrastive_pairs(self.train_data)
        test_contrastive = self.create_contrastive_pairs(self.test_data)
        
        with open(output_path / 'train_contrastive.json', 'w', encoding='utf-8') as f:
            json.dump(train_contrastive, f, ensure_ascii=False, indent=2)
            
        with open(output_path / 'test_contrastive.json', 'w', encoding='utf-8') as f:
            json.dump(test_contrastive, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        stats = {
            'total_samples': len(self.qa_data),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'avg_question_length': np.mean([len(item['question'].split()) for item in self.qa_data]),
            'avg_answer_length': np.mean([len(item['answer'].split()) for item in self.qa_data]),
            'question_types': {}
        }
        
        # Count question types
        for item in self.qa_data:
            q_type = item.get('type_question', 'unknown')
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
        
        return stats
    
    def print_data_info(self) -> None:
        """Print dataset information"""
        stats = self.get_data_statistics()
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Train samples: {stats['train_samples']}")
        print(f"Test samples: {stats['test_samples']}")
        print(f"Average question length: {stats['avg_question_length']:.1f} words")
        print(f"Average answer length: {stats['avg_answer_length']:.1f} words")
        
        print("\nQuestion types:")
        for q_type, count in stats['question_types'].items():
            percentage = (count / stats['total_samples']) * 100
            print(f"  {q_type}: {count} ({percentage:.1f}%)")
        
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader
    data_loader = EmbeddingDataLoader(
        data_path=config['general']['data_path']
    )
    
    # Print dataset info
    data_loader.print_data_info()
    
    # Create datasets for different training approaches
    train_dataset = data_loader.create_sentence_transformer_dataset(data_loader.train_data)
    triplets = data_loader.create_triplet_dataset(data_loader.train_data)
    
    print(f"\nCreated {len(train_dataset)} training examples")
    print(f"Created {len(triplets)} triplets")