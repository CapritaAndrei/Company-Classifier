"""
Threshold and selection strategies for classification.
"""

import pandas as pd


class TopKSelector:
    """Handles top-K selection for classification results."""
    
    def __init__(self, k=5):
        self.k = k
    
    def select_top_k_per_company(self, scores_df, k=None):
        """Select top K labels per company based on similarity scores."""
        if k is None:
            k = self.k
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # Group by company and select top K
        top_k_results = scores_df.groupby('company_id', group_keys=False).apply(
            lambda x: x.nlargest(k, 'similarity_score')
        )
        
        return top_k_results
    
    def select_top_k_per_company_per_model(self, scores_df, k=None):
        """Select top K labels per company per model."""
        if k is None:
            k = self.k
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # Group by company and model, then select top K
        top_k_results = scores_df.groupby(['company_id', 'embedding_model'], group_keys=False).apply(
            lambda x: x.nlargest(k, 'similarity_score')
        )
        
        return top_k_results
    
    def apply_threshold_after_topk(self, top_k_df, threshold):
        """Apply threshold filtering after top-K selection."""
        return top_k_df[top_k_df['similarity_score'] >= threshold] 