import os
import math
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Sample knowledge base corpus
CORPUS = [
    {"id": "doc1", "title": "Apple iPhone 15 Pro", "content": "Features the A17 Pro chip, titanium design, action button, and USB-C port."},
    {"id": "doc2", "title": "MacBook Air M3", "content": "Lightweight laptop powered by Apple M3 silicon with 18 hours of battery life."},
    {"id": "doc3", "title": "Sony WH-1000XM5", "content": "Industry-leading noise-canceling wireless headphones with dual processors."},
    {"id": "doc4", "title": "Samsung Galaxy S24 Ultra", "content": "Flagship smartphone featuring Galaxy AI, Snapdragon 8 Gen 3, and S Pen support."},
    {"id": "doc5", "title": "Apple Fruit Health Benefits", "content": "Apples are rich in fiber, vitamin C, and antioxidants. Great for digestion and heart health."},
    {"id": "doc6", "title": "Dell XPS 15", "content": "High-performance laptop with Intel Core i9, NVIDIA RTX 4070, and OLED 3.5K display."}
]

# -------------------------------------------------------------------
# 1. Sparse Search (BM25 Keyword Matching)
# -------------------------------------------------------------------
tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
bm25 = BM25Okapi(tokenized_corpus)

def search_sparse(query: str, top_k: int = 5):
    """Performs exact term matching via BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Pair scores with documents and sort descending
    scored_docs = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# -------------------------------------------------------------------
# 2. Dense Search (OpenAI Embeddings)
# -------------------------------------------------------------------
def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """Generates dense vector representation using OpenAI."""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Pre-compute document embeddings
doc_embeddings = {doc["id"]: get_embedding(f"{doc['title']} {doc['content']}") for doc in CORPUS}

def search_dense(query: str, top_k: int = 5):
    """Performs semantic similarity search via OpenAI embeddings."""
    query_vector = get_embedding(query)
    
    scored_docs = []
    for doc in CORPUS:
        doc_vector = doc_embeddings[doc["id"]]
        sim = cosine_similarity(query_vector, doc_vector)
        scored_docs.append((doc, sim))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# -------------------------------------------------------------------
# 3. Reciprocal Rank Fusion (RRF)
# -------------------------------------------------------------------
def reciprocal_rank_fusion(rankings_list: list[list[dict]], k: int = 60):
    """
    Fuses multiple ranked lists using the RRF algorithm.
    RRF Score = SUM( 1 / (k + rank) ) across all retrieval channels.
    """
    rrf_scores = defaultdict(float)
    doc_map = {}

    for rankings in rankings_list:
        for rank, doc in enumerate(rankings, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort documents by accumulated RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]

# -------------------------------------------------------------------
# 4. Reranking (LLM as Cross-Encoder)
# -------------------------------------------------------------------
def rerank_with_llm(query: str, candidate_docs: list[dict], top_k: int = 3):
    """
    Uses an LLM cross-encoder pass to score exact relevance 
    for the top RRF candidate pool.
    """
    scored_candidates = []

    for doc in candidate_docs:
        prompt = f"""You are a search relevance evaluator.
Score the relevance of the following Document to the User Query on a scale of 0 to 10.
Respond ONLY with a single numeric value between 0 and 10.

Query: "{query}"
Document Title: "{doc['title']}"
Document Content: "{doc['content']}"

Relevance Score:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
        except ValueError:
            score = 0.0

        scored_candidates.append((doc, score))

    # Sort by LLM relevance score descending
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_k]

# -------------------------------------------------------------------
# Pipeline Execution & Demonstration
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Query contains both keyword signals ('apple') and conceptual intent ('high performance laptop')
    user_query = "apple high performance laptop"

    print(f"\n=== USER QUERY: '{user_query}' ===\n")

    # Step 1 & 2: Perform Hybrid Retrieval
    sparse_results = search_sparse(user_query, top_k=4)
    dense_results = search_dense(user_query, top_k=4)

    print("--- [1] Sparse Search Results (BM25) ---")
    for i, doc in enumerate(sparse_results, 1):
        print(f"  {i}. [{doc['id']}] {doc['title']}")

    print("\n--- [2] Dense Search Results (OpenAI Embeddings) ---")
    for i, doc in enumerate(dense_results, 1):
        print(f"  {i}. [{doc['id']}] {doc['title']}")

    # Step 3: Combine with RRF
    rrf_fused_candidates = reciprocal_rank_fusion([sparse_results, dense_results], k=60)

    print("\n--- [3] Reciprocal Rank Fusion (RRF Combined Candidates) ---")
    for i, doc in enumerate(rrf_fused_candidates, 1):
        print(f"  {i}. [{doc['id']}] {doc['title']}")

    # Step 4: Rerank top candidates with OpenAI Cross-Encoder logic
    final_reranked = rerank_with_llm(user_query, rrf_fused_candidates[:4], top_k=2)

    print("\n--- [4] Final Reranked Results (LLM Cross-Encoder) ---")
    for i, (doc, score) in enumerate(final_reranked, 1):
        print(f"  Rank {i} (Score: {score}/10): [{doc['id']}] {doc['title']}")
        print(f"          Snippet: {doc['content']}")