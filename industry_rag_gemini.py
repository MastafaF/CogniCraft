import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import os
import google.generativeai as genai

# ==============================================================================
# --- 1. RAG System Core Functions ---
# ==============================================================================

def initialize_search_services(documents, model):
    """
    Initializes FAISS index and BM25 model from the documents.
    """
    print("Initializing FAISS and BM25 services for Terminology Assistant...")
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 model initialized.")

    embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} vectors.")
    
    return bm25, faiss_index

def hybrid_search(query, model, bm25, faiss_index, documents, source_lang, target_lang, top_k=3):
    """
    Performs a hybrid search to retrieve relevant documents.
    """
    print(f"\nPerforming hybrid search for: '{query}'")
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding, k=top_k)
    
    rrf_scores = {}
    k = 60
    
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, doc_idx in enumerate(bm25_ranked_indices[:20]):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    for rank, doc_idx in enumerate(indices[0]):
        if doc_idx != -1:
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + (1 / (k + rank))

    sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    final_results = []
    print("\n--- Top Retrieved Documents ---")
    for doc_idx in sorted_doc_indices[:top_k]:
        source_text = documents.iloc[doc_idx][source_lang]
        target_text = documents.iloc[doc_idx][target_lang]
        final_results.append({
            "source": source_text,
            "target": target_text,
            "score": rrf_scores[doc_idx]
        })
        print(f"  EN: {source_text}")
        print(f"  ES: {target_text}\n")
        
    return final_results

def generate_response(query, retrieved_docs):
    """
    Generates a response using the Gemini LLM, anchored on the retrieved documents.
    """
    print("--- Generating Final Response with LLM ---")
    
    # Configure the Gemini client
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return "Error: API key not configured. Please set the GEMINI_API_KEY environment variable."

    # Format the retrieved documents for the prompt
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1} (EN): {doc['source']}\n"
        context += f"Document {i+1} (ES): {doc['target']}\n\n"

    # Create the prompt
    prompt = f"""
    You are a translation assistant for a company like Apple. Your task is to help a translator by answering their question based *only* on the provided official documents.
    The user is asking a question about how to translate a concept. Provide a clear, concise answer and show the approved translation examples from the documents.

    **Translator's Question:**
    "{query}"

    **Official Documents:**
    {context}

    **Your Answer:**
    """
    
    # --- Call Gemini API using the official Python client ---
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    try:
        response = model.generate_content(prompt)
        generated_text = response.text
        print(generated_text)
        return generated_text
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return f"Error: Failed to generate content from the model. Details: {e}"


# ==============================================================================
# --- 2. Scenario-Specific Setup and Execution ---
# ==============================================================================
if __name__ == "__main__":
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
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    source_documents = tm_df['english'].tolist()
    bm25_model, faiss_model = initialize_search_services(source_documents, model)

    # --- Full RAG Pipeline ---
    
    # Query 1: Keyword-heavy query
    translator_query_1 = "How do I use Face ID to unlock my phone?"
    # Not: top_k is set to 3 for this scenario
    # This query is expected to retrieve documents that contain the keyword "Face ID" and related
    retrieved_docs_1 = hybrid_search(
        query=translator_query_1, model=model, bm25=bm25_model, faiss_index=faiss_model,
        documents=tm_df, source_lang='english', target_lang='spanish'
    )
    generate_response(translator_query_1, retrieved_docs_1)
    
    print("\n" + "="*50 + "\n")

    # Query 2: Semantic-heavy query
    # This query is more about the concept of using a phone for payments, which may not be keyword-heavy
    translator_query_2 = "A new way to pay with your phone."
    retrieved_docs_2 = hybrid_search(
        query=translator_query_2, model=model, bm25=bm25_model, faiss_index=faiss_model,
        documents=tm_df, source_lang='english', target_lang='spanish'
    )
    generate_response(translator_query_2, retrieved_docs_2)

    print("\n" + "="*50 + "\n")

    # Query 3: Slogan query
    # This query is about the company's slogan, which is a specific piece of information
    # that may not be directly related to product features but is important for brand identity
    translator_query_3 = "What is the main company slogan?"
    retrieved_docs_3 = hybrid_search(
        query=translator_query_3, model=model, bm25=bm25_model, faiss_index=faiss_model,
        documents=tm_df, source_lang='english', target_lang='spanish'
    )
    generate_response(translator_query_3, retrieved_docs_3)
