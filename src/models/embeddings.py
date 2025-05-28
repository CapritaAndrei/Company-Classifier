"""
Embedding model management for company classification.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages embedding models and generation."""
    
    def __init__(self, models_config, device):
        self.models_config = models_config
        self.device = device
        self.loaded_models = {}
        
    def load_models(self):
        """Load all embedding models."""
        print(f"Loading embedding models...")
        for model_key, model_name in self.models_config.items():
            print(f"  - Loading Embedding Model '{model_key}': {model_name}")
            try:
                if model_name == 'BAAI/bge-m3':
                    self.loaded_models[model_key] = SentenceTransformer(
                        model_name, device=self.device, trust_remote_code=True
                    )
                else:
                    self.loaded_models[model_key] = SentenceTransformer(
                        model_name, device=self.device
                    )
                print(f"    Embedding model '{model_key} ({model_name})' loaded successfully.")
            except Exception as e:
                print(f"    Error loading embedding model '{model_key} ({model_name})': {e}")
        return self.loaded_models
    
    def get_embeddings(self, texts_list, model_obj, batch_size=32):
        """Generate embeddings for a list of texts."""
        if not texts_list or not all(isinstance(t, str) for t in texts_list):
            return np.array([])
        try:
            all_embeddings = model_obj.encode(
                texts_list, 
                convert_to_tensor=True, 
                show_progress_bar=False, 
                batch_size=batch_size
            )
            return all_embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error during get_embeddings: {e}")
            return np.array([])
    
    def generate_embeddings_for_dataframe(self, df, text_column, model_key):
        """Generate embeddings for a dataframe column."""
        if model_key not in self.loaded_models:
            print(f"Model {model_key} not loaded.")
            return df
        
        model_obj = self.loaded_models[model_key]
        embeddings = self.get_embeddings(df[text_column].tolist(), model_obj)
        
        if len(embeddings) == len(df):
            df[f'{model_key}_embedding'] = list(embeddings)
            print(f"    Stored embeddings in column '{model_key}_embedding'. Shape example: {embeddings[0].shape if len(embeddings) > 0 and hasattr(embeddings[0], 'shape') else 'N/A'}")
        else:
            print(f"    Warning: Mismatch in length between dataframe ({len(df)}) and generated embeddings ({len(embeddings)}) for {model_key}.")
        
        return df 