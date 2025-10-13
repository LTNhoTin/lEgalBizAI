"""
Evaluation utilities for embedding model finetuning
Implements: Recall@k, MRR@k, nDCG@k, Context Precision
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    """Evaluator for embedding models with multiple metrics"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        
    def recall_at_k(self, relevant_docs: List[List[int]], retrieved_docs: List[List[int]], k: int) -> float:
        """
        Calculate Recall@k
        
        Args:
            relevant_docs: List of lists containing relevant document IDs for each query
            retrieved_docs: List of lists containing retrieved document IDs for each query
            k: Number of top documents to consider
            
        Returns:
            Recall@k score (0-1)
        """
        if len(relevant_docs) != len(retrieved_docs):
            raise ValueError("Number of queries must match between relevant and retrieved docs")
            
        total_recall = 0.0
        valid_queries = 0
        
        for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
            if not rel_docs:  # Skip queries with no relevant documents
                continue
                
            ret_docs_k = ret_docs[:k]
            relevant_retrieved = len(set(rel_docs) & set(ret_docs_k))
            recall = relevant_retrieved / len(rel_docs)
            total_recall += recall
            valid_queries += 1
            
        return total_recall / valid_queries if valid_queries > 0 else 0.0
    
    def mrr_at_k(self, relevant_docs: List[List[int]], retrieved_docs: List[List[int]], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank@k (MRR@k)
        
        Args:
            relevant_docs: List of lists containing relevant document IDs for each query
            retrieved_docs: List of lists containing retrieved document IDs for each query
            k: Number of top documents to consider
            
        Returns:
            MRR@k score (0-1)
        """
        if len(relevant_docs) != len(retrieved_docs):
            raise ValueError("Number of queries must match between relevant and retrieved docs")
            
        total_rr = 0.0
        valid_queries = 0
        
        for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
            if not rel_docs:  # Skip queries with no relevant documents
                continue
                
            ret_docs_k = ret_docs[:k]
            rr = 0.0
            
            for rank, doc_id in enumerate(ret_docs_k, 1):
                if doc_id in rel_docs:
                    rr = 1.0 / rank
                    break
                    
            total_rr += rr
            valid_queries += 1
            
        return total_rr / valid_queries if valid_queries > 0 else 0.0
    
    def ndcg_at_k(self, relevant_docs: List[List[int]], retrieved_docs: List[List[int]], 
                  relevance_scores: List[List[float]], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k (nDCG@k)
        
        Args:
            relevant_docs: List of lists containing relevant document IDs for each query
            retrieved_docs: List of lists containing retrieved document IDs for each query
            relevance_scores: List of lists containing relevance scores for each document
            k: Number of top documents to consider
            
        Returns:
            nDCG@k score (0-1)
        """
        if len(relevant_docs) != len(retrieved_docs) or len(relevant_docs) != len(relevance_scores):
            raise ValueError("All input lists must have the same length")
            
        total_ndcg = 0.0
        valid_queries = 0
        
        for rel_docs, ret_docs, rel_scores in zip(relevant_docs, retrieved_docs, relevance_scores):
            if not rel_docs:  # Skip queries with no relevant documents
                continue
                
            # Calculate DCG@k
            dcg = 0.0
            ret_docs_k = ret_docs[:k]
            
            for rank, doc_id in enumerate(ret_docs_k, 1):
                if doc_id in rel_docs:
                    doc_idx = rel_docs.index(doc_id)
                    relevance = rel_scores[doc_idx] if doc_idx < len(rel_scores) else 1.0
                    dcg += relevance / np.log2(rank + 1)
            
            # Calculate IDCG@k (Ideal DCG)
            ideal_scores = sorted(rel_scores, reverse=True)[:k]
            idcg = sum(score / np.log2(rank + 2) for rank, score in enumerate(ideal_scores))
            
            # Calculate nDCG@k
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg
            valid_queries += 1
            
        return total_ndcg / valid_queries if valid_queries > 0 else 0.0
    
    def context_precision(self, relevant_docs: List[List[int]], retrieved_docs: List[List[int]], k: int) -> float:
        """
        Calculate Context Precision@k
        Percentage of retrieved documents that are actually relevant
        
        Args:
            relevant_docs: List of lists containing relevant document IDs for each query
            retrieved_docs: List of lists containing retrieved document IDs for each query
            k: Number of top documents to consider
            
        Returns:
            Context Precision@k score (0-1)
        """
        if len(relevant_docs) != len(retrieved_docs):
            raise ValueError("Number of queries must match between relevant and retrieved docs")
            
        total_precision = 0.0
        valid_queries = 0
        
        for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
            if not ret_docs:  # Skip queries with no retrieved documents
                continue
                
            ret_docs_k = ret_docs[:k]
            relevant_retrieved = len(set(rel_docs) & set(ret_docs_k))
            precision = relevant_retrieved / len(ret_docs_k)
            total_precision += precision
            valid_queries += 1
            
        return total_precision / valid_queries if valid_queries > 0 else 0.0
    
    def evaluate_all_metrics(self, relevant_docs: List[List[int]], retrieved_docs: List[List[int]], 
                           relevance_scores: List[List[float]] = None) -> Dict[str, Dict[int, float]]:
        """
        Evaluate all metrics for all k values
        
        Args:
            relevant_docs: List of lists containing relevant document IDs for each query
            retrieved_docs: List of lists containing retrieved document IDs for each query
            relevance_scores: List of lists containing relevance scores (optional, defaults to 1.0)
            
        Returns:
            Dictionary containing all metrics for all k values
        """
        if relevance_scores is None:
            # Default relevance scores to 1.0 for all relevant documents
            relevance_scores = [[1.0] * len(rel_docs) for rel_docs in relevant_docs]
            
        results = {
            'recall': {},
            'mrr': {},
            'ndcg': {},
            'context_precision': {}
        }
        
        for k in self.k_values:
            results['recall'][k] = self.recall_at_k(relevant_docs, retrieved_docs, k)
            results['mrr'][k] = self.mrr_at_k(relevant_docs, retrieved_docs, k)
            results['ndcg'][k] = self.ndcg_at_k(relevant_docs, retrieved_docs, relevance_scores, k)
            results['context_precision'][k] = self.context_precision(relevant_docs, retrieved_docs, k)
            
        return results
    
    def print_evaluation_report(self, results: Dict[str, Dict[int, float]], 
                              targets: Dict[str, float] = None) -> None:
        """
        Print a formatted evaluation report
        
        Args:
            results: Results from evaluate_all_metrics
            targets: Target values for each metric
        """
        if targets is None:
            targets = {
                'recall': 0.85,
                'mrr': 0.8,
                'ndcg': 0.8,
                'context_precision': 0.85
            }
        
        print("\n" + "="*60)
        print("EMBEDDING MODEL EVALUATION REPORT")
        print("="*60)
        
        for metric_name, metric_results in results.items():
            print(f"\n{metric_name.upper()}@k:")
            print("-" * 30)
            
            target_value = targets.get(metric_name, 0.0)
            
            for k, value in metric_results.items():
                status = "✓ PASS" if value >= target_value else "✗ FAIL"
                print(f"  {metric_name}@{k:2d}: {value:.4f} (Target: {target_value:.2f}) {status}")
        
        print("\n" + "="*60)

def prepare_evaluation_data(qa_data: List[Dict], chunk_embeddings: np.ndarray, 
                          query_embeddings: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Prepare evaluation data from QA dataset and embeddings
    
    Args:
        qa_data: List of QA dictionaries with chunk_ids
        chunk_embeddings: Embeddings for all chunks
        query_embeddings: Embeddings for all queries
        
    Returns:
        Tuple of (relevant_docs, retrieved_docs)
    """
    relevant_docs = []
    retrieved_docs = []
    
    for i, qa_item in enumerate(qa_data):
        # Get relevant chunk IDs for this query
        relevant_chunk_ids = qa_item.get('chunk_ids', [])
        relevant_docs.append(relevant_chunk_ids)
        
        # Calculate similarity and get top-k retrieved documents
        query_emb = query_embeddings[i:i+1]
        similarities = cosine_similarity(query_emb, chunk_embeddings)[0]
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        retrieved_docs.append(sorted_indices.tolist())
    
    return relevant_docs, retrieved_docs