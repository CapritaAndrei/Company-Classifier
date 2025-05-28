import sys
import os
import time
import torch
import pandas as pd

# Path setup
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_path = os.path.dirname(src_dir)
if project_root_for_path not in sys.path:
    sys.path.insert(0, project_root_for_path)

# NLTK setup
import nltk
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_nltk_data_path = os.path.join(project_root, 'nltk_data')
nltk.data.path = [project_nltk_data_path]

# Import configurations
from .config import (
    COMPANY_DATA_FILE, TAXONOMY_FILE, SAMPLE_SIZE,
    NLI_MODEL_NAME, EMBEDDING_MODELS_CONFIG,
    USE_CACHE, PREPROCESSED_COMPANIES_CACHE_FILE, PREPROCESSED_TAXONOMY_CACHE_FILE,
    COMPANIES_WITH_EMBEDDINGS_CACHE_FILE, TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE
)

# Import modules
from .data.loader import load_data
from .data.preprocessor import DataPreprocessor
from .models.embeddings import EmbeddingManager
from .models.nli import NLIManager
from .classification.similarity import SimilarityClassifier
from .classification.thresholds import TopKSelector

# Configuration
FORCE_CPU = True
TOP_N_EMBEDDING_MATCHES = 15


def main():
    """Main execution function."""
    print("--- Starting Company Classification Process ---")
    overall_start_time = time.time()
    
    # Initialize components
    preprocessor = DataPreprocessor(use_cache=USE_CACHE)
    similarity_classifier = SimilarityClassifier()
    top_k_selector = TopKSelector(k=TOP_N_EMBEDDING_MATCHES)
    
    # Device setup
    if FORCE_CPU:
        device = torch.device("cpu")
        print("Forcing CPU usage.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load or retrieve data
    companies_df, taxonomy_df = load_or_process_data(preprocessor)
    
    if companies_df.empty or taxonomy_df.empty:
        print("Critical error: No data available. Exiting.")
        sys.exit(1)
    
    # Step 2: Apply sampling if configured
    companies_df, taxonomy_df = apply_sampling(companies_df, taxonomy_df)
    
    # Step 3: Load models and generate embeddings
    embedding_manager = EmbeddingManager(EMBEDDING_MODELS_CONFIG, device)
    loaded_models = embedding_manager.load_models()
    
    if not loaded_models:
        print("No embedding models loaded. Exiting.")
        sys.exit(1)
    
    companies_df, taxonomy_df = generate_or_load_embeddings(
        companies_df, taxonomy_df, embedding_manager, loaded_models, preprocessor
    )
    
    # Step 4: Perform classification
    all_scores = perform_classification(
        companies_df, taxonomy_df, loaded_models, similarity_classifier
    )
    
    # Step 5: Process results
    process_results(all_scores, top_k_selector)
    
    # End timing
    total_time = time.time() - overall_start_time
    print(f"\n--- TOTAL EXECUTION TIME: {total_time:.2f} seconds ---")


def load_or_process_data(preprocessor):
    """Load data from cache or process from scratch."""
    # Try loading from embeddings cache first
    companies_df = preprocessor.load_from_cache(COMPANIES_WITH_EMBEDDINGS_CACHE_FILE)
    taxonomy_df = preprocessor.load_from_cache(TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE)
    
    if companies_df is not None and taxonomy_df is not None:
        print(f"Loaded data with embeddings from cache: {len(companies_df)} companies, {len(taxonomy_df)} labels")
        return companies_df, taxonomy_df
    
    # Try loading from preprocessed cache
    companies_df = preprocessor.load_from_cache(PREPROCESSED_COMPANIES_CACHE_FILE)
    taxonomy_df = preprocessor.load_from_cache(PREPROCESSED_TAXONOMY_CACHE_FILE)
    
    if companies_df is not None and taxonomy_df is not None:
        print(f"Loaded preprocessed data from cache: {len(companies_df)} companies, {len(taxonomy_df)} labels")
        return companies_df, taxonomy_df
    
    # Load raw data and preprocess
    print("\n--- Loading and preprocessing raw data ---")
    companies_df, taxonomy_df = load_data(COMPANY_DATA_FILE, TAXONOMY_FILE)
    
    if companies_df.empty or taxonomy_df.empty:
        return companies_df, taxonomy_df
    
    # Preprocess
    companies_df = preprocessor.preprocess_companies(companies_df)
    taxonomy_df = preprocessor.preprocess_taxonomy(taxonomy_df)
    
    # Save to cache
    preprocessor.save_to_cache(companies_df, PREPROCESSED_COMPANIES_CACHE_FILE)
    preprocessor.save_to_cache(taxonomy_df, PREPROCESSED_TAXONOMY_CACHE_FILE)
    
    return companies_df, taxonomy_df


def apply_sampling(companies_df, taxonomy_df):
    """Apply sampling if configured."""
    if SAMPLE_SIZE is not None:
        print(f"\n--- Applying SAMPLE_SIZE: Processing first {SAMPLE_SIZE} companies ---")
        companies_df = companies_df.head(SAMPLE_SIZE).copy()
        print(f"After sampling: {len(companies_df)} companies, {len(taxonomy_df)} taxonomy labels.")
    else:
        print(f"\n--- Processing ALL {len(companies_df)} companies against {len(taxonomy_df)} taxonomy entries ---")
    
    return companies_df, taxonomy_df


def generate_or_load_embeddings(companies_df, taxonomy_df, embedding_manager, loaded_models, preprocessor):
    """Generate embeddings or load from cache."""
    # Check if embeddings already exist
    embedding_cols = [col for col in companies_df.columns if '_embedding' in col]
    if embedding_cols:
        print("Embeddings already present in data.")
        return companies_df, taxonomy_df
    
    print("\n--- Generating Embeddings ---")
    
    # Generate company embeddings
    if 'company_full_text_structured' in companies_df.columns:
        for model_key in loaded_models.keys():
            companies_df = embedding_manager.generate_embeddings_for_dataframe(
                companies_df, 'company_full_text_structured', model_key
            )
    
    # Generate taxonomy embeddings
    if 'taxonomy_full_text_structured' in taxonomy_df.columns:
        for model_key in loaded_models.keys():
            taxonomy_df = embedding_manager.generate_embeddings_for_dataframe(
                taxonomy_df, 'taxonomy_full_text_structured', model_key
            )
    
    # Save to cache
    preprocessor.save_to_cache(companies_df, COMPANIES_WITH_EMBEDDINGS_CACHE_FILE)
    preprocessor.save_to_cache(taxonomy_df, TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE)
    
    return companies_df, taxonomy_df


def perform_classification(companies_df, taxonomy_df, loaded_models, similarity_classifier):
    """Perform similarity-based classification."""
    print("\n--- Performing Classification ---")
    embedding_start_time = time.time()
    
    all_scores = []
    
    for model_key in loaded_models.keys():
        model_start_time = time.time()
        print(f"\n--- Processing with model: {model_key} ---")
        
        scores = similarity_classifier.classify_all_scores(companies_df, taxonomy_df, model_key)
        all_scores.extend(scores)
        
        model_duration = time.time() - model_start_time
        print(f"--- Model {model_key} processing time: {model_duration:.2f} seconds ---")
    
    total_embedding_duration = time.time() - embedding_start_time
    print(f"\n--- TOTAL CLASSIFICATION TIME: {total_embedding_duration:.2f} seconds ---")
    
    return all_scores


def process_results(all_scores, top_k_selector):
    """Process and display classification results."""
    if not all_scores:
        print("No classification results generated.")
        return
    
    scores_df = pd.DataFrame(all_scores)
    
    print(f"\n--- Top {TOP_N_EMBEDDING_MATCHES} Matches (per company, per model) ---")
    top_matches = top_k_selector.select_top_k_per_company_per_model(scores_df)
    
    if not top_matches.empty:
        print(top_matches.to_string(index=False))
    else:
        print("No results to display.")


if __name__ == "__main__":
    main() 