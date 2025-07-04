import torch
import pandas as pd
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np

# --- Configuration ---
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
INDEX_NAME = "cognicraft_index"

# ==============================================================================
# --- PART 1: OPENSEARCH-BASED HYBRID SEARCH ---
# ==============================================================================

def initialize_opensearch_services():
    """Initializes OpenSearch client and the sentence embedding model."""
    print("Initializing OpenSearch services...")
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT, 'scheme': 'http'}]
    )
    if not client.ping():
        raise ValueError("Connection to OpenSearch failed!")
    print("OpenSearch connection successful.")
    model = SentenceTransformer('intfloat/e5-small-v2')
    print("Sentence Transformer model loaded.")
    return client, model

def init_faiss_model_only():
    """Initializes sentence embedding model."""
    print("Initializing OpenSearch services...")
    model = SentenceTransformer('intfloat/e5-small-v2')
    print("Sentence Transformer model loaded.")
    return model


def create_opensearch_index(client):
    """Creates an OpenSearch index with a specific mapping for text and embeddings."""
    if not client.indices.exists(index=INDEX_NAME):
        print(f"Creating OpenSearch index '{INDEX_NAME}'...")
        settings = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 384}
                }
            }
        }
        client.indices.create(index=INDEX_NAME, body=settings)
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

def index_data_in_opensearch(client, model, data):
    """Indexes data into OpenSearch after generating embeddings."""
    print("Indexing data into OpenSearch...")
    actions = []
    for text in data['content']:
        embedding = model.encode(text, convert_to_tensor=True).tolist()
        actions.append({
            "_index": INDEX_NAME,
            "_source": {"text": text, "embedding": embedding}
        })
    helpers.bulk(client, actions)
    print(f"Successfully indexed {len(actions)} documents in OpenSearch.")

def hybrid_search_opensearch(client, model, query, bm25_weight=0.5, semantic_weight=0.5):
    """Performs a hybrid search using OpenSearch with score normalization."""
    print(f"\nPerforming OpenSearch hybrid search for: '{query}'")
    
    # --- Step 1: Get results from both search types ---
    bm25_query = {"query": {"match": {"text": query}}}
    bm25_results = client.search(index=INDEX_NAME, body=bm25_query)
    
    query_embedding = model.encode(query, convert_to_tensor=True).tolist()
    semantic_query = {"query": {"knn": {"embedding": {"vector": query_embedding, "k": 5}}}}
    semantic_results = client.search(index=INDEX_NAME, body=semantic_query)

    # --- Step 2: Normalize and combine scores ---
    combined_scores = {}
    
    # Helper for Min-Max normalization
    def normalize_scores(hits):
        scores = [hit['_score'] for hit in hits]
        if not scores:
            return {}
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score: # Avoid division by zero
            return {hit['_id']: 1.0 for hit in hits}
        
        normalized = {hit['_id']: (hit['_score'] - min_score) / (max_score - min_score) for hit in hits}
        return normalized

    bm25_normalized_scores = normalize_scores(bm25_results['hits']['hits'])
    semantic_normalized_scores = normalize_scores(semantic_results['hits']['hits'])

    # Combine using the normalized scores and weights
    for doc_id, score in bm25_normalized_scores.items():
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (score * bm25_weight)
    
    for doc_id, score in semantic_normalized_scores.items():
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (score * semantic_weight)

    # --- Step 3: Sort and return final results ---
    sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    final_results = []
    for doc_id, score in sorted_docs[:5]:
        doc = client.get(index=INDEX_NAME, id=doc_id)
        final_results.append({"text": doc['_source']['text'], "combined_score": score})
    return final_results

# ==============================================================================
# --- PART 2: FAISS + BM25-BASED HYBRID SEARCH ---
# ==============================================================================

def initialize_faiss_bm25_services(documents, model):
    """
    Initializes FAISS index and BM25 model from the documents.

    Args:
        documents (list of str): The documents to be indexed.
            Example: ["This is the first document.", "This is the second one."]
        model: The sentence-transformer model instance.
    """
    print("\nInitializing FAISS and BM25 services...")
    
    # --- BM25 Setup ---
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 model initialized.")

    # --- FAISS Setup ---
    embeddings = model.encode(documents, convert_to_tensor=True).cpu().numpy()
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} vectors.")
    
    return bm25, faiss_index

def hybrid_search_faiss_bm25(query, model, bm25, faiss_index, documents):
    """Performs a hybrid search using FAISS and BM25, combining with Reciprocal Rank Fusion."""
    print(f"\nPerforming FAISS+BM25 hybrid search for: '{query}'")

    # --- BM25 Search (Keyword) ---
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # --- Semantic Search (Embeddings) ---
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding, k=5)
    
    # --- Combine and Re-rank Results (Reciprocal Rank Fusion - RRF) ---
    rrf_scores = {}
    k = 60 # RRF constant
    
    # Process BM25 results
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, doc_idx in enumerate(bm25_ranked_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    # Process FAISS results
    for rank, doc_idx in enumerate(indices[0]):
        if doc_idx != -1:
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    # Sort documents by combined RRF score
    sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    final_results = []
    for doc_idx in sorted_doc_indices[:5]:
        final_results.append({
            "text": documents[doc_idx],
            "combined_score": rrf_scores[doc_idx]
        })
    return final_results

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    generic_data = {
        'content': [
            "The sky is blue and the sun is bright.",
            "Apples are a type of fruit that grow on trees.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is a branch of computer science.",
            "Paris is the capital city of France."
        ]
    }
    documents_list = generic_data['content']

    # --- Run OpenSearch Approach ---
    # print("--- 1. RUNNING OPENSEARCH HYBRID SEARCH ---")
    # os_client, sentence_model = initialize_opensearch_services()
    sentence_model = init_faiss_model_only()
    # create_opensearch_index(os_client)
    # index_data_in_opensearch(os_client, sentence_model, generic_data)
    # test_query = "What color is the sky?"
    test_query = "What is the capital of France?"
    # os_results = hybrid_search_opensearch(os_client, sentence_model, test_query)
    # print("\n--- OpenSearch Results (with Normalized Scores) ---")
    # for res in os_results:
    #     print(f"Score: {res['combined_score']:.4f} | Text: {res['text']}")

    # --- Run FAISS + BM25 Approach ---
    print("\n\n--- 2. RUNNING FAISS + BM25 HYBRID SEARCH ---")
    bm25_model, faiss_model = initialize_faiss_bm25_services(documents_list, sentence_model)
    faiss_results = hybrid_search_faiss_bm25(test_query, sentence_model, bm25_model, faiss_model, documents_list)
    print("\n--- FAISS + BM25 Results (with RRF Scores) ---")
    for res in faiss_results:
        print(f"Score: {res['combined_score']:.4f} | Text: {res['text']}")

    print("\n\n--- COMPARISON COMPLETE ---")