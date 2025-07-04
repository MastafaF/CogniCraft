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

### 1. OpenSearch Approach: Normalized Weighted Linear Combination

The OpenSearch method first normalizes the scores from each search system to a common [0, 1] scale using **Min-Max Normalization**. This prevents one system's scoring scale from unfairly dominating the other.


For each set of results (BM25 and Semantic), the scores are normalized:
```math
S_{\text{norm}} = \frac{S - S_{\text{min}}}{S_{\text{max}} - S_{\text{min}}}
```


Then, a **weighted linear combination** is applied to the normalized scores. For a given document `d` and query `q`, the combined score is:

$$
S_{\text{combined}}(d, q) = (w_{\text{bm25}} \times S_{\text{norm,bm25}}) + (w_{\text{semantic}} \times S_{\text{norm,semantic}})
$$

Where:
- `$S_{\text{norm}}$` is the normalized score from each respective system (BM25 or semantic).
- `$w_{\text{bm25}}$` and `$w_{\text{semantic}}$` are the weights assigned to prioritize one search method over the other.
- The weights should sum to 1 (e.g., $w_{\text{bm25}} = 0.5$, $w_{\text{semantic}} = 0.5$ for equal weighting).
- The final combined score determines the ranking of each document.

### 2. FAISS + BM25 Approach: Reciprocal Rank Fusion (RRF)

The in-memory approach uses **Reciprocal Rank Fusion (RRF)**, which combines results based on their rank in each list, not their raw scores.

The RRF score for a document `d` is calculated as:

$$
S_{\text{RRF}}(d) = \sum_{i \in \text{systems}} \frac{1}{k + \text{rank}_i(d)}
$$

Where:
-   `$\text{rank}_i(d)$` is the rank of document `d` in the results from system `i` (either BM25 or FAISS).
-   `$k$` is a constant (set to 60 in our script) used to reduce the impact of high ranks.

### RRF vs. Weighted Approach: A Practical Example

The difference between these two methods is critical to understand. Let's use the query **"What is the capital of France?"** to see why.

**Results using RRF:**

Score: 0.0333 | Text: Paris is the capital city of France.
Score: 0.0328 | Text: The sky is blue and the sun is bright.
... (other scores are very close)


**Analysis:**

1.  **Why is the top result correct?**
    The document "Paris is the capital city of France" was ranked #1 by BM25 (keyword match on "France") and #1 by FAISS (semantic match). Its RRF score is calculated as:
    -   BM25 rank contribution: `1 / (60 + 0) = 0.0167`
    -   FAISS rank contribution: `1 / (60 + 0) = 0.0167`
    -   **Total Score: `0.0334`**

2.  **Why are the other scores so close?**
    RRF gives points based on **rank, not relevance**. Even an irrelevant document will get a score based on its rank in each list. For example, if "The sky is blue..." was ranked 5th by BM25 (`rank=4`) and 2nd by FAISS (`rank=1`), its score would be `1/(60+4) + 1/(60+1) = 0.0320`. The scores are numerically close because the formula is designed to give a slight edge to documents that are ranked consistently high across different systems.

**Key Comparison:**

| Feature | Weighted Approach (OpenSearch) | Reciprocal Rank Fusion (RRF) |
| :--- | :--- | :--- |
| **How it Works** | Combines the *actual scores* from each system. | Combines the *ranks* from each system. |
| **Main Advantage** | Intuitive weights give direct control. | **No need for score normalization.** It works brilliantly even if the underlying systems have vastly different score ranges. This makes it very robust. |
| **Main Disadvantage**| **Requires careful normalization.** If scores aren't normalized, one system can easily dominate the other, making the weights meaningless. | The final scores are less intuitive and often numerically close. The final sorted order is what matters, not the absolute score. |

For combining heterogeneous systems like keyword search and vector search, **RRF is often a more practical and robust choice** as it sidesteps the complex problem of score normalization.

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
5.  **View results:**
    The script will print the results of both hybrid search approaches, showing how they retrieve and rank documents based on the query.