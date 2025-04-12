import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, logging
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import time

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
NLTK_RESOURCES = ['stopwords', 'wordnet']
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 100
SIMILARITY_THRESHOLD = 0.28 # Adjusted threshold
SEMANTIC_WEIGHT = 0.65      # Adjusted weight
KEYPHRASE_WEIGHT = 0.35     # Adjusted weight
TOP_N_MATCHES = 3

# --- Utility Functions ---
def download_nltk_resources():
    """Download necessary NLTK resources if not already present."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)

def preprocess_text(text):
    """Clean and standardize text."""
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    return " ".join(filtered_words)

def get_taxonomy_phrases(label):
    """Extract meaningful single words and 2-word phrases from taxonomy labels."""
    label = label.lower()
    stop_words = set(stopwords.words('english')) | {'and', 'of', 'for', 'the', 'in', 'to'}
    words = [w for w in re.findall(r'\b\w+\b', label) if w not in stop_words]

    phrases = set(words) # Start with single words
    # Add 2-word phrases
    if len(words) >= 2:
        for i in range(len(words) - 1):
            phrases.add(words[i] + " " + words[i+1])

    # Very specific common insurance/business phrases (optional, can be expanded)
    known_phrases = ["general liability", "professional liability", "property insurance", "workers compensation", "cyber liability"]
    for kp in known_phrases:
        if kp in label:
            phrases.add(kp)

    return list(phrases)


# --- Core Logic Functions ---
def load_data(company_file, taxonomy_file):
    """Load company and taxonomy data."""
    print("Loading data...")
    companies_df = pd.read_csv(company_file)
    taxonomy_df = pd.read_excel(taxonomy_file)
    print(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels.")
    return companies_df, taxonomy_df

def initialize_model(model_name):
    """Initialize the Sentence Transformer model."""
    print(f"Loading SentenceBERT model: {model_name}...")
    return SentenceTransformer(model_name)

def create_taxonomy_data(taxonomy_df, model):
    """Create embeddings and extract phrases for taxonomy labels."""
    print("Processing taxonomy labels...")
    taxonomy_embeddings_list = []
    taxonomy_phrases_list = []

    for label in tqdm(taxonomy_df['label']):
        phrases = get_taxonomy_phrases(label)
        taxonomy_phrases_list.append(phrases)

        # Create descriptive context for embedding
        # Focus on type of business/service described by the label
        context_keywords = [p for p in phrases if len(p.split()) == 1][:5] # Use top single keywords for context
        context = f"Insurance context: Business operations related to {label}. Keywords: {', '.join(context_keywords)}."
        embedding = model.encode(context)
        taxonomy_embeddings_list.append(embedding)

    return np.array(taxonomy_embeddings_list), taxonomy_phrases_list

def create_company_representation(row):
    """Create a structured and weighted text representation for a company."""
    desc = preprocess_text(row['description'])
    tags = preprocess_text(row['business_tags'])
    sector = preprocess_text(row.get('sector', ''))
    category = preprocess_text(row.get('category', ''))
    niche = preprocess_text(row.get('niche', ''))

    # Structure the text representation, giving more weight to specific fields
    representation = (
        f"description: {desc}. "
        f"tags: {tags}. {tags}. " # Repeat tags for weight
        f"classification: sector {sector}, category {category}, niche {niche}. {niche}. " # Repeat niche
        f"industry context: {sector} {category} {niche}."
    )
    return representation

def classify_companies(companies_df, taxonomy_df, taxonomy_embeddings_np, taxonomy_phrases_list, model):
    """Classify companies against the taxonomy using a hybrid approach."""
    num_companies = len(companies_df)
    all_matches = []
    taxonomy_labels = taxonomy_df['label'].tolist() # Get labels for matching output

    print(f"Classifying {num_companies} companies...")

    for start_idx in tqdm(range(0, num_companies, BATCH_SIZE)):
        end_idx = min(start_idx + BATCH_SIZE, num_companies)
        batch_df = companies_df.iloc[start_idx:end_idx]

        # 1. Generate Company Embeddings for the batch
        company_texts = [create_company_representation(row) for _, row in batch_df.iterrows()]
        company_embeddings_np = model.encode(company_texts)

        # 2. Calculate Semantic Similarity
        semantic_matrix = cosine_similarity(company_embeddings_np, taxonomy_embeddings_np)

        # 3. Calculate Key Phrase Matching Score
        phrase_matrix = np.zeros((len(batch_df), len(taxonomy_labels)))
        for i, company_text in enumerate(company_texts):
            company_words = set(company_text.split()) # For faster lookup
            for j, label_phrases in enumerate(taxonomy_phrases_list):
                matches = sum(1 for phrase in label_phrases if phrase in company_text) # Check if phrase exists in company text
                # Normalize score by number of phrases in the label
                phrase_matrix[i, j] = matches / max(1, len(label_phrases))

        # 4. Combine Scores
        combined_matrix = (SEMANTIC_WEIGHT * semantic_matrix) + (KEYPHRASE_WEIGHT * phrase_matrix)

        # 5. Determine Matches based on Threshold
        for i, (original_idx, _) in enumerate(batch_df.iterrows()): # Use original index
            scores = combined_matrix[i]
            top_indices = np.argsort(scores)[::-1][:TOP_N_MATCHES] # Get top N indices

            matches = []
            for label_idx in top_indices:
                score = scores[label_idx]
                if score >= SIMILARITY_THRESHOLD:
                    matches.append((taxonomy_labels[label_idx], score))

            all_matches.append({
                'company_idx': original_idx, # Store original DataFrame index
                'matches': matches
            })

    return all_matches

def generate_output(companies_df, classification_results, output_base_name="classified_companies"):
    """Generate the final CSV output files."""
    print("Generating output files...")
    output_df = companies_df.copy()
    output_df['insurance_label'] = ""
    output_df['insurance_label_with_scores'] = ""

    matched_count = 0
    total_labels_assigned = 0

    results_map = {result['company_idx']: result['matches'] for result in classification_results}

    for idx, row in output_df.iterrows():
        matches = results_map.get(idx, [])
        if matches:
            matched_count += 1
            total_labels_assigned += len(matches)
            output_df.at[idx, 'insurance_label'] = '; '.join([m[0] for m in matches])
            output_df.at[idx, 'insurance_label_with_scores'] = '; '.join([f"{m[0]} ({m[1]:.2f})" for m in matches])

    # Reorder columns for readable output
    readable_cols = [
        'description', 'business_tags', 'insurance_label', 'insurance_label_with_scores',
        'sector', 'category', 'niche'
    ]
    readable_df = output_df[readable_cols]
    readable_output_file = f"{output_base_name}_readable.csv"
    readable_df.to_csv(readable_output_file, index=False)
    print(f"Readable results saved to '{readable_output_file}'")

    # Create summary
    percent_matched = (matched_count / len(companies_df)) * 100 if len(companies_df) > 0 else 0
    avg_labels = total_labels_assigned / matched_count if matched_count > 0 else 0
    summary = pd.DataFrame([{
        'total_companies': len(companies_df),
        'companies_with_labels': matched_count,
        'percentage_matched': f"{percent_matched:.1f}%",
        'avg_labels_per_matched_company': f"{avg_labels:.2f}"
    }])
    summary_output_file = f"{output_base_name}_summary.csv"
    summary.to_csv(summary_output_file, index=False)
    print(f"Summary statistics saved to '{summary_output_file}'")

    # Save the original dataframe with just the 'insurance_label' column added (as requested by task)
    main_output_file = f"{output_base_name}.csv"
    output_df.to_csv(main_output_file, index=False)
    print(f"Main results (with insurance_label column) saved to '{main_output_file}'")


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    download_nltk_resources()
    companies_df, taxonomy_df = load_data('ml_insurance_challenge.csv', 'insurance_taxonomy.xlsx')
    model = initialize_model(MODEL_NAME)
    taxonomy_embeddings_np, taxonomy_phrases_list = create_taxonomy_data(taxonomy_df, model)
    classification_results = classify_companies(companies_df, taxonomy_df, taxonomy_embeddings_np, taxonomy_phrases_list, model)
    generate_output(companies_df, classification_results) # Pass original df

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")