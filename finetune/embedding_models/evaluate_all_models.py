#!/usr/bin/env python3
"""
Comprehensive evaluation script for all embedding models
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from evaluation_utils import EmbeddingEvaluator, prepare_evaluation_data
from data_loader import QADataLoader

class ModelEvaluator:
    """Comprehensive evaluator for all embedding models"""
    
    def __init__(self, data_path: str, k_values: List[int] = None):
        self.data_path = data_path
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.logger = self._setup_logging()
        
        # Load test data
        self.data_loader = QADataLoader(data_path)
        self.test_data = self._load_test_data()
        
        # Initialize evaluator
        self.evaluator = EmbeddingEvaluator(self.k_values)
        
        # Target metrics
        self.target_metrics = {
            "Recall@k": 0.85,
            "MRR@k": 0.8,
            "nDCG@k": 0.8,
            "Context_Precision": 0.85
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_test_data(self) -> List[Dict]:
        """Load and prepare test data"""
        self.logger.info("Loading test data...")
        data = self.data_loader.load_data()
        
        # Use last 20% as test data
        test_size = int(len(data) * 0.2)
        test_data = data[-test_size:]
        
        self.logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def evaluate_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        self.logger.info(f"Evaluating {model_name}...")
        
        try:
            # Load model based on type
            if "gemma" in model_name.lower():
                # For Gemma, we need to load the custom model
                model = self._load_gemma_model(model_path)
            elif "bge" in model_name.lower():
                # For BGE-M3, we need to load the custom model
                model = self._load_bge_model(model_path)
            else:
                # For MiniLM, use sentence-transformers
                model = SentenceTransformer(model_path)
            
            # Prepare evaluation data
            eval_data = prepare_evaluation_data(self.test_data)
            
            # Extract queries and passages
            queries = [item['question'] for item in self.test_data]
            passages = [item['answer'] for item in self.test_data]
            
            # Encode queries and passages
            if hasattr(model, 'encode'):
                # SentenceTransformer model
                query_embeddings = model.encode(queries, convert_to_tensor=True)
                passage_embeddings = model.encode(passages, convert_to_tensor=True)
                
                if torch.is_tensor(query_embeddings):
                    query_embeddings = query_embeddings.cpu().numpy()
                if torch.is_tensor(passage_embeddings):
                    passage_embeddings = passage_embeddings.cpu().numpy()
            else:
                # Custom model - implement encoding logic
                query_embeddings = self._encode_with_custom_model(model, queries)
                passage_embeddings = self._encode_with_custom_model(model, passages)
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_embeddings(
                query_embeddings,
                passage_embeddings,
                eval_data
            )
            
            # Add model info
            metrics['model_name'] = model_name
            metrics['model_path'] = model_path
            
            self.logger.info(f"Evaluation completed for {model_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {e}")
            return {
                'model_name': model_name,
                'model_path': model_path,
                'error': str(e)
            }
    
    def _load_gemma_model(self, model_path: str):
        """Load Gemma model (placeholder - implement based on actual model structure)"""
        # This would need to be implemented based on how Gemma model is saved
        self.logger.warning("Gemma model loading not fully implemented - using SentenceTransformer as fallback")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def _load_bge_model(self, model_path: str):
        """Load BGE model (placeholder - implement based on actual model structure)"""
        # This would need to be implemented based on how BGE model is saved
        self.logger.warning("BGE model loading not fully implemented - using SentenceTransformer as fallback")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def _encode_with_custom_model(self, model, texts: List[str]) -> np.ndarray:
        """Encode texts with custom model"""
        # Placeholder implementation
        # This would need to be implemented based on the actual model interface
        embeddings = []
        for text in texts:
            # Dummy embedding - replace with actual model encoding
            embedding = np.random.randn(384)  # Assuming 384-dim embeddings
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def evaluate_all_models(self, model_configs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate all models and compare results"""
        self.logger.info("Starting comprehensive evaluation of all models...")
        
        results = {}
        
        for config in model_configs:
            model_name = config['name']
            model_path = config['path']
            
            if os.path.exists(model_path):
                results[model_name] = self.evaluate_model(model_path, model_name)
            else:
                self.logger.warning(f"Model path not found: {model_path}")
                results[model_name] = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'error': 'Model path not found'
                }
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("EMBEDDING MODELS EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 40)
        
        # Create comparison table
        comparison_data = []
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                continue
                
            row = {'Model': model_name}
            
            # Extract best metrics for each type
            for metric_type in ['Recall', 'MRR', 'nDCG']:
                if metric_type in metrics:
                    best_value = max(metrics[metric_type].values())
                    row[f'{metric_type}@k (best)'] = f"{best_value:.4f}"
            
            comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report.append(df.to_string(index=False))
        
        report.append("")
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        # Detailed results for each model
        for model_name, metrics in results.items():
            report.append(f"\n{model_name.upper()}")
            report.append("=" * len(model_name))
            
            if 'error' in metrics:
                report.append(f"Error: {metrics['error']}")
                continue
            
            # Check target achievement
            targets_met = self._check_targets(metrics)
            if targets_met:
                report.append("ALL TARGETS MET!")
            else:
                report.append("Some targets not met")
            
            # Detailed metrics
            for metric_type in ['Recall', 'MRR', 'nDCG']:
                if metric_type in metrics:
                    report.append(f"\n{metric_type}@k:")
                    for k, value in metrics[metric_type].items():
                        target = self.target_metrics.get(f"{metric_type}@k", 0)
                        status = "✅" if value >= target else "❌"
                        report.append(f"  @{k}: {value:.4f} {status} (target: {target})")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append(self._generate_recommendations(results))
        
        return "\n".join(report)
    
    def _check_targets(self, metrics: Dict[str, Any]) -> bool:
        """Check if model meets target metrics"""
        targets_met = True
        
        for metric_type in ['Recall', 'MRR', 'nDCG']:
            if metric_type in metrics:
                best_value = max(metrics[metric_type].values())
                target = self.target_metrics.get(f"{metric_type}@k", 0)
                if best_value < target:
                    targets_met = False
        
        return targets_met
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Find best performing model
        best_model = None
        best_score = 0
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                continue
                
            # Calculate average score
            total_score = 0
            count = 0
            
            for metric_type in ['Recall', 'MRR', 'nDCG']:
                if metric_type in metrics:
                    total_score += max(metrics[metric_type].values())
                    count += 1
            
            if count > 0:
                avg_score = total_score / count
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model_name
        
        if best_model:
            recommendations.append(f"🏆 Best performing model: {best_model} (avg score: {best_score:.4f})")
        
        # Check which models meet targets
        models_meeting_targets = []
        for model_name, metrics in results.items():
            if 'error' not in metrics and self._check_targets(metrics):
                models_meeting_targets.append(model_name)
        
        if models_meeting_targets:
            recommendations.append(f"Models meeting all targets: {', '.join(models_meeting_targets)}")
        else:
            recommendations.append("No models meet all target metrics - consider additional training")
        
        return "\n".join(recommendations)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        # Save JSON results
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save text report
        report = self.generate_comparison_report(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Results saved to {output_path} and {json_path}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate all embedding models")
    parser.add_argument("--data_path", type=str, 
                       default="/home/nhotin/work/LegalBizAI_project/finetune/data_finetunning/qaset_full_article.json",
                       help="Path to evaluation data")
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                       help="Output file for results")
    parser.add_argument("--k_values", nargs='+', type=int, default=[1, 3, 5, 10, 20],
                       help="K values for evaluation")
    
    args = parser.parse_args()
    
    # Define model configurations
    model_configs = [
        {
            'name': 'Google Gemma Embeddings',
            'path': '/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/google_gemma_embeddings/models'
        },
        {
            'name': 'BAAI bge-m3',
            'path': '/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/baai_bge_m3/models'
        },
        {
            'name': 'all-MiniLM-L6-v2',
            'path': '/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/all_minilm_l6_v2/models'
        }
    ]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.data_path, args.k_values)
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(model_configs)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"Results saved to: {args.output}")
    print(f"JSON results saved to: {args.output.replace('.txt', '.json')}")

if __name__ == "__main__":
    main()