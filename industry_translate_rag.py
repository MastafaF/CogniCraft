import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

# ==============================================================================
# --- 1. RAG System Core Functions ---
# ==============================================================================

def initialize_search_services(documents, model):
    """
    Initializes FAISS index and BM25 model from the documents.

    Args:
        documents (list of str): The source sentences to be indexed.
            Example: ["This is the first document.", "This is the second one."]
        model: The sentence-transformer model instance.
    """
    print("Initializing FAISS and BM25 services for Terminology Assistant...")
    
    # --- BM25 Setup (for keyword search) ---
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 model initialized.")

    # --- FAISS Setup (for semantic search) ---
    embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} vectors.")
    
    return bm25, faiss_index

def hybrid_search(query, model, bm25, faiss_index, documents, source_lang, target_lang, top_k=5):
    """
    Performs a hybrid search using FAISS and BM25, combining with Reciprocal Rank Fusion.

    Args:
        query (str): The source text query from the translator.
        model: The sentence-transformer model.
        bm25: The initialized BM25 model.
        faiss_index: The initialized FAISS index.
        documents (pd.DataFrame): The DataFrame containing the translation memory.
        source_lang (str): The column name for the source language in the DataFrame.
        target_lang (str): The column name for the target language in the DataFrame.
        top_k (int): The number of results to return.
    """
    print(f"\nPerforming hybrid search for: '{query}'")

    # --- BM25 Search (Keyword) ---
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # --- Semantic Search (Embeddings) ---
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding, k=top_k)
    
    # --- Combine and Re-rank Results (Reciprocal Rank Fusion - RRF) ---
    rrf_scores = {}
    k = 60 # RRF constant
    
    # Process BM25 results
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, doc_idx in enumerate(bm25_ranked_indices[:20]): # Consider top 20 from BM25
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    # Process FAISS results
    for rank, doc_idx in enumerate(indices[0]):
        if doc_idx != -1:
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    # Sort documents by combined RRF score
    sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    # Fetch the full translation pairs for the top results
    final_results = []
    print("\n--- Top Terminology Matches ---")
    for doc_idx in sorted_doc_indices[:top_k]:
        source_text = documents.iloc[doc_idx][source_lang]
        target_text = documents.iloc[doc_idx][target_lang]
        final_results.append({
            "source": source_text,
            "target": target_text,
            "score": rrf_scores[doc_idx]
        })
        print(f"Score: {rrf_scores[doc_idx]:.4f}")
        print(f"  EN: {source_text}")
        print(f"  ES: {target_text}\n")
        
    return final_results

# ==============================================================================
# --- 2. Scenario-Specific Setup and Execution ---
# ==============================================================================
if __name__ == "__main__":
    # --- Step 1: Load a "Translation Memory" (TM) ---
    # This is our toy dataset of previously approved translations.
    # In a real scenario, this would be a large database.
    translation_data = {
        'english': [
            "Unlock your iPhone with Face ID.",
            "Face ID is a secure and private way to authenticate.",
            "The new MacBook Pro features a stunning Liquid Retina XDR display.",
            "Our iconic slogan is 'Think Different'.",
            "You can use Apple Pay to make secure purchases in stores.",
            "How to set up facial recognition on your device.",
            "The slogan that defined our brand is 'Think Different'.",
            "Apple Pay is accepted at millions of locations worldwide.",
            "The Liquid Retina XDR display offers true-to-life color."
        ],
        'spanish': [
            "Desbloquea tu iPhone con Face ID.",
            "Face ID es una forma segura y privada de autenticación.",
            "El nuevo MacBook Pro tiene una espectacular pantalla Liquid Retina XDR.",
            "Nuestro lema icónico es 'Think Different'.",
            "Puedes usar Apple Pay para hacer compras seguras en tiendas.",
            "Cómo configurar el reconocimiento facial en tu dispositivo.",
            "El lema que definió nuestra marca es 'Think Different'.",
            "Apple Pay se acepta en millones de establecimientos en todo el mundo.",
            "La pantalla Liquid Retina XDR ofrece colores realistas."
        ]
    }
    tm_df = pd.DataFrame(translation_data)
    
    # --- Step 2: Initialize the RAG system ---
    # We use a multilingual model suitable for this task
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # We index the English source text
    source_documents = tm_df['english'].tolist()
    bm25_model, faiss_model = initialize_search_services(source_documents, model)

    # --- Step 3: Simulate Translator Queries ---
    
    # Query 1: A translator needs to translate a sentence with a key product name.
    # BM25 should excel here.
    translator_query_1 = "How do I use Face ID to unlock my phone?"
    results_1 = hybrid_search(
        query=translator_query_1,
        model=model,
        bm25=bm25_model,
        faiss_index=faiss_model,
        documents=tm_df,
        source_lang='english',
        target_lang='spanish'
    )
    
    # Query 2: A translator has a sentence about a concept, not an exact product name.
    # Semantic search should help find the relevant term.
    translator_query_2 = "A new way to pay with your phone."
    results_2 = hybrid_search(
        query=translator_query_2,
        model=model,
        bm25=bm25_model,
        faiss_index=faiss_model,
        documents=tm_df,
        source_lang='english',
        target_lang='spanish'
    )

    # Query 3: A translator needs to find the official translation of the company slogan.
    # Both systems should strongly agree on this.
    translator_query_3 = "What is the main company slogan?"
    results_3 = hybrid_search(
        query=translator_query_3,
        model=model,
        bm25=bm25_model,
        faiss_index=faiss_model,
        documents=tm_df,
        source_lang='english',
        target_lang='spanish'
    )

