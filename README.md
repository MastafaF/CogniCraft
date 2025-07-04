# CogniCraft: A Hybrid RAG System

CogniCraft is a Python-based project designed to demonstrate and compare different approaches for building a hybrid Retrieval-Augmented Generation (RAG) system. It serves as a learning tool to understand the practical differences between using an integrated search platform like OpenSearch versus a manual, in-memory combination of libraries like FAISS and BM25.

## Project Goal

The primary goal of this project is to build a RAG system that leverages both traditional keyword-based search (BM25) and modern semantic search (vector embeddings). This hybrid approach aims to improve retrieval accuracy by finding documents that are relevant both lexically (matching keywords) and semantically (matching meaning).

## Approaches Compared

This repository implements two distinct architectures for hybrid search:

1.  **OpenSearch-based Hybrid Search:**
    * Uses a single, persistent OpenSearch instance to handle data storage, BM25 keyword search, and k-NN vector search.
    * Represents a more robust, production-oriented architecture.

2.  **FAISS + BM25 Hybrid Search:**
    * Uses in-memory libraries: `faiss-cpu` for vector search and `rank_bm25` for keyword search.
    * Represents a lightweight, script-based approach suitable for rapid prototyping and experimentation.

## How the Hybrid Search Works (Mathematical Explanation)

The hybrid search functionality in this project combines the results from keyword and semantic search to produce a single, re-ranked list of documents.

### 1. OpenSearch Approach: Weighted Linear Combination

The OpenSearch method uses a **weighted linear combination** to merge scores.

For a given document `d` and query `q`, the combined score is calculated as:

$$
S_{\text{combined}}(d, q) = (w_{\text{bm25}} \times S_{\text{bm25}}(d, q)) + (w_{\text{semantic}} \times S_{\text{semantic}}(d, q))
$$

Where:
-   `S_bm25(d, q)` is the keyword relevance score from OpenSearch's BM25 algorithm.
-   `S_semantic(d, q)` is the vector similarity score from OpenSearch's k-NN search.
-   `w_bm25` and `w_semantic` are the weights you can assign to prioritize one search method over the other.

### 2. FAISS + BM25 Approach: Reciprocal Rank Fusion (RRF)

The in-memory approach uses **Reciprocal Rank Fusion (RRF)**, which combines results based on their rank in each list, not their raw scores.

The RRF score for a document `d` is calculated as:

$$
S_{\text{RRF}}(d) = \sum_{i \in \text{systems}} \frac{1}{k + \text{rank}_i(d)}
$$

Where:
-   `rank_i(d)` is the rank of document `d` in the results from system `i` (either BM25 or FAISS).
-   `k` is a constant (set to 60 in our script) used to reduce the impact of high ranks.

This method is effective because it doesn't require score normalization and is less sensitive to the different scoring scales of the underlying systems.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/CogniCraft.git](https://github.com/YOUR_USERNAME/CogniCraft.git)
    cd CogniCraft
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start OpenSearch (for the first part of the script):**
    ```bash
    docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:latest
    ```

4.  **Run the script:**
    ```bash
    python cognicraft_rag.py
    ```
