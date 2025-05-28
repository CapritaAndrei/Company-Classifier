import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # src -> project_root

COMPANY_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'input', 'ml_insurance_challenge.csv')
TAXONOMY_FILE = os.path.join(PROJECT_ROOT, 'data', 'input', 'insurance_taxonomy.xlsx')
OUTPUT_BASE_NAME = os.path.join(PROJECT_ROOT, 'output', 'classified_companies_main') # Assuming output goes to output dir

# --- Cache Configuration ---
USE_CACHE = True # Master switch for using cached intermediate data
PREPROCESSED_COMPANIES_CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_companies.pkl')
PREPROCESSED_TAXONOMY_CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_taxonomy.pkl')
COMPANIES_WITH_EMBEDDINGS_CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'companies_with_embeddings.pkl')
TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'taxonomy_with_embeddings.pkl')
NLI_RESULTS_CACHE_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nli_results_cache.pkl')

SAMPLE_SIZE = None # Process all companies for full performance test

# --- NLI Configuration ---
NLI_MODEL_NAME = 'facebook/bart-large-mnli'
NLI_ENTAILMENT_THRESHOLD = 0.8
NLI_BATCH_SIZE = 1  # <<< ADDED: Number of companies to process in NLI before saving intermediate results

# --- Embedding Configuration ---
EMBEDDING_MODELS_CONFIG = {
    'mini_lm': 'all-MiniLM-L6-v2',
    'bge_m3': 'BAAI/bge-m3'
}
EMBEDDING_SIMILARITY_THRESHOLD = 0.7

