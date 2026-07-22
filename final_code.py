#%%
import os
import re
import pprint
import numpy as np
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 1. DATA LOADING & STRUCTURING
# ==============================================================================
DATA_DIR = os.path.join("data", "nutrition")

CUSTOM_METADATA = {
    "01_overview_of_nutrition.md": {
        "title": "Overview of Nutrition: Small Animals",
        "source": "MSD Veterinary Manual ",
        "category": "Nutrition",
        "last_updated": "2024-09",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/overview-of-nutrition-small-animals "
    },
    "02_nutritional_req.md": {
        "title": "Nutritional Requirements of Small Animals",
        "source": "MSD Veterinary Manual ",
        "category": "Nutritional Requirements",
        "last_updated": "2024-09",
        "target_species": ["Dogs", "Cats"],
        "url_reference": " https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/nutritional-requirements-of-small-animals"
    },
    "03_nutrition_disease_managment.md": {
        "title": "Nutrition in Disease Management in Small Animals",
        "source": "MSD Veterinary Manual",
        "category": " Management and Nutrition",
        "last_updated": "2025-06",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/nutrition-in-disease-management-in-small-animals"
    },
    "04_feeding_practices.md": {
        "title": "Feeding Practices in Small Animals",
        "source_": "MSD Veterinary Manual",
        "category": " Management and Nutrition",
        "last_updated": "2024-10",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/feeding-practices-in-small-animals "
    },
    "05_foods_managment.md": {
        "title": "Dog and Cat Foods",
        "source": "MSD Veterinary Manual",
        "category": " Dog and Cat Foods",
        "last_updated": "2025-06",
        "target_species": ["Dogs", "Cats"],
        "url_reference": "https://www.msdvetmanual.com/management-and-nutrition/nutrition-small-animals/dog-and-cat-foods"
    }
}

loader = DirectoryLoader(
    DATA_DIR,
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
raw_documents = loader.load()

documents = []
for index, doc in enumerate(raw_documents):
    file_name = os.path.basename(doc.metadata["source"])
    file_metadata = CUSTOM_METADATA.get(file_name, {})
    
    formatted_doc = {
        "document_id": index,
        "title": file_metadata.get("title", ""),
        "source": file_metadata.get("source", file_metadata.get("source_", "")),
        "category": file_metadata.get("category", ""),
        "last_updated": file_metadata.get("last_updated", ""),
        "target_species": file_metadata.get("target_species", []),
        "url_reference": file_metadata.get("url_reference", ""),
        "text": doc.page_content
    }
    documents.append(formatted_doc)

#%%
# 2. GROUND TRUTH & UPDATED QUERIES DATAFRAME WITH QUERY_ID
# ==============================================================================
ground_truth = {
    "What is the nutritional requirements of dogs ?": [0],
    "what are the body condition score scales?  ":[0],
    "what is the meaning of malnutrition?":[0],
    "how protein requirements of dogs and cats vary vary ?":[1],
    "what are the different life stages in nutrient requirements?" :[1],
    "what is the importance of water to pets?":[1],
    "what are the signs of protein deficiency?":[1],
    "How many times per day should puppies between weaning and 6 months old be fed?" : [2],
    "What are the main factors for developing obesity in dogs?" : [2],
    "What is the pros and cons of protein?" : [1,2],
    "What are the three most common food allergens in domestic dogs?" : [3],
    "what is the caloric requirements of lactating queen?":[3],
    "What is the best feeding regimen to adult dogs?" :[3],
    "what is the impact of overfeeding" : [2,3],
    "How does excessive fat tissue actively contribute to chronic joint inflammation and metabolic diseases?" :[3],
    "Which amino acid deficiency causes dilated cardiomyopathy and central retinal degeneration in cats?":[4],
    "Why are plant-based ingredients incapable of fulfilling a cat's organic requirement for Vitamin A and fatty acids?" :[4],
    "are dog foods satisfactory for cats?" : [4],
    "How does an excessive intake of a single mineral negatively impact the utilization of other essential elements?": [4],
    "Write down the exact exponential formula used to calculate a pet's Resting Energy Requirement (RER), and explain how this calculation be modified (by what specific percentage) when starting a weight loss program for an overweight feline." :[3,4],
    "Why is it incorrect to view adult maintenance as a single, unchanging life stage as a pet ages, and how must the digestibility of macronutrients be modified when feeding older felines compared to middle-aged ones?" : [2,4]
}

queries_df = pd.DataFrame({
    "query": list(ground_truth.keys()),
    "relevant_document_ids": list(ground_truth.values())
})

# Added query_id generation mapped as q1, q2, q3...
queries_df.insert(0, "query_id", [f"q{i+1}" for i in range(len(queries_df))])

queries_df

#%%
# 3. CLEANING, TEXT SPLITTING AND CHUNKING
# ==============================================================================
def clean_final_chunk(text):
    text = text.replace('\\n', '\n').lower()
    text = re.sub(r"[^a-z0-9\s\-%\+/=\><]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunk_rows = []
for doc in documents:
    md_header_splits = markdown_splitter.split_text(doc["text"])
    final_chunks = text_splitter.split_documents(md_header_splits)
    
    for chunk_index, chunk in enumerate(final_chunks):
        clean_chunk_content = clean_final_chunk(chunk.page_content)
        
        h1 = chunk.metadata.get("header_1", "")
        h2 = chunk.metadata.get("header_2", "")
        h3 = chunk.metadata.get("header_3", "")
        headers_context = f"{h1} {h2} {h3}".strip()

        chunk_rows.append({
            "chunk_id": f"doc{doc['document_id']}_chunk{chunk_index}",
            "document_id": doc["document_id"],
            "title": doc["title"],
            "source": doc["source"],
            "category": doc["category"],
            "last_updated": doc["last_updated"],
            "target_species": ", ".join(doc["target_species"]) if isinstance(doc["target_species"], list) else doc["target_species"],
            "url_reference": doc["url_reference"],
            "chunk_index": chunk_index,
            "chunk_text": clean_chunk_content,
            "search_text": f"Context: {headers_context} | Title: {doc['title']} | Category: {doc['category']} | Content: {clean_chunk_content}"
        })

chunks_df = pd.DataFrame(chunk_rows)

chunks_df

#%%
# 4. RETRIEVERS SETUP
# ==============================================================================
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(chunks_df["search_text"])

def retrieve_top_k_tfidf(query, k=3):
    query_vector = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"
    return results[["retriever", "chunk_id", "document_id", "title", "category", "source", "score", "chunk_text"]].reset_index(drop=True)

def simple_tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

tokenized_chunks = [simple_tokenize(text) for text in chunks_df["search_text"]]
bm25 = BM25Okapi(tokenized_chunks)

def retrieve_top_k_bm25(query, k=3):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "BM25"
    return results[["retriever", "chunk_id", "document_id", "title", "category", "source", "score", "chunk_text"]].reset_index(drop=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_texts = chunks_df["search_text"].tolist()
chunk_embeddings = model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True)

def retrieve_top_k_semantic(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"
    return results[["retriever", "chunk_id", "document_id", "title", "category", "source", "score", "chunk_text"]].reset_index(drop=True)

def retrieve_top_k_hybrid(query, k=3, alpha=0.5):
    tokenized_query = simple_tokenize(query)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    
    def min_max_normalize(scores):
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min == 0:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    lexical_norm = min_max_normalize(bm25_scores)
    semantic_norm = min_max_normalize(semantic_scores)
    
    hybrid_scores = alpha * lexical_norm + (1 - alpha) * semantic_norm
    ranking = np.argsort(hybrid_scores)[::-1][:k]
    
    results = chunks_df.iloc[ranking].copy()
    results["score"] = hybrid_scores[ranking]
    results["retriever"] = "Hybrid"
    return results[["retriever", "chunk_id", "document_id", "title", "category", "source", "score", "chunk_text"]].reset_index(drop=True)

#%%
def precision_at_k(retrieved_ids, relevant_ids, k):
    hits = [doc_id for doc_id in retrieved_ids[:k] if doc_id in relevant_ids]
    return len(hits) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)

def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return int(len(hits) > 0)

def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in set(relevant_ids):
            return 1 / rank
    return 0.0

def evaluate_retriever_on_df(retriever_name, retrieval_function, eval_df, k=3):
    rows = []
    for _, row_data in eval_df.iterrows():
        query_text = row_data["query"]
        results = retrieval_function(query_text, k)
        retrieved_doc_ids = [int(x) for x in results["document_id"].tolist()]
        raw_relevant = row_data["relevant_document_ids"]
        
        relevant_ids = raw_relevant if isinstance(raw_relevant, list) else [raw_relevant]
        relevant_ids = [int(x) for x in relevant_ids]
        
        rows.append({
            "retriever": retriever_name,
            "query": query_text,
            f"precision@{k}": precision_at_k(retrieved_doc_ids, relevant_ids, k),
            f"recall@{k}": recall_at_k(retrieved_doc_ids, relevant_ids, k),
            f"hit_rate@{k}": hit_rate_at_k(retrieved_doc_ids, relevant_ids, k),
            "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant_ids),
        })
    return pd.DataFrame(rows)
#%%
#Performance Evaluation Dashboard & Live Sample Target Matching
K_EVAL = 3

tfidf_eval_results = evaluate_retriever_on_df("TF-IDF", retrieve_top_k_tfidf, queries_df, k=K_EVAL)
bm25_eval_results = evaluate_retriever_on_df("BM25", retrieve_top_k_bm25, queries_df, k=K_EVAL)
semantic_eval_results = evaluate_retriever_on_df("Semantic (Embeddings)", retrieve_top_k_semantic, queries_df, k=K_EVAL)
hybrid_eval_results = evaluate_retriever_on_df("Hybrid Search", lambda q, k: retrieve_top_k_hybrid(q, k, alpha=0.5), queries_df, k=K_EVAL)

all_results = pd.concat([tfidf_eval_results, bm25_eval_results, semantic_eval_results, hybrid_eval_results])

comparison_dashboard = all_results.groupby("retriever").agg({
    f"precision@{K_EVAL}": "mean",
    f"recall@{K_EVAL}": "mean",
    f"hit_rate@{K_EVAL}": "mean",
    "reciprocal_rank": "mean"
}).rename(columns={"reciprocal_rank": "Mean Reciprocal Rank (MRR)"}).reset_index()

print("=== RETRIEVER PERFORMANCE COMPARISON ===")
print(comparison_dashboard.to_string(index=False))
print("\n" + "="*80 + "\n")

target_query_id = "q10"

test_query = queries_df.loc[queries_df["query_id"] == target_query_id, "query"].values[0]

print(f"=== LIVE SAMPLE TEST USING HYBRID RETRIEVER ===")
print(f"Query ID: {target_query_id}")
print(f"Testing Query: '{test_query}'\n")

sample_results = retrieve_top_k_hybrid(test_query, k=2)
sample_results
#%%
# 5. CONTEXT BUILDING MODULE (INTEGRATED & SYNCED)
# ==============================================================================
def build_context_package(query_id, retriever="hybrid", k=6, word_budget=300, max_chunks=3):
    matched_queries = queries_df.loc[queries_df["query_id"] == query_id]
    
    if matched_queries.empty:
        raise ValueError(f"Query ID '{query_id}' not found in queries_df. Available IDs: {queries_df['query_id'].tolist()}")
        
    query_row = matched_queries.iloc[0]
    query_text = query_row["query"]

    if retriever == "hybrid":
        candidates = retrieve_top_k_hybrid(query_text, k=k).copy()
    elif retriever == "semantic":
        candidates = retrieve_top_k_semantic(query_text, k=k).copy()
    elif retriever == "bm25":
        candidates = retrieve_top_k_bm25(query_text, k=k).copy()
    else:
        candidates = retrieve_top_k_tfidf(query_text, k=k).copy()

    candidates = candidates.sort_values(by=["score"], ascending=[False]).reset_index(drop=True)

    selected_rows = []
    seen_texts = set()
    used_words = 0

    for _, row in candidates.iterrows():
        normalized_text = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized_text in seen_texts:
            continue

        chunk_words = len(row["chunk_text"].split())
        if selected_rows and used_words + chunk_words > word_budget:
            continue

        selected_rows.append(row.to_dict())
        seen_texts.add(normalized_text)
        used_words += chunk_words

        if len(selected_rows) >= max_chunks:
            break

    selected_df = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame()
    context_blocks = []

    for idx, row in enumerate(selected_rows, start=1):
        context_blocks.append(
            f"[Source {idx}] {row['title']} | category={row['category']} | source={row['source']} | score={row['score']:.4f}\n"
            f"{row['chunk_text']}"
        )

    return {
        "query_row": query_row,
        "candidates": candidates,
        "selected_chunks_df": selected_df,
        "used_words": used_words,
        "context_text": "\n\n".join(context_blocks),
    }

target_query_id = "q14"
context_package_demo = build_context_package(target_query_id, retriever="hybrid", k=6)

print(f"=== CONTEXT GENERATION FOR QUERY ID: {target_query_id} ===")
print(f"Testing Query: '{context_package_demo['query_row']['query']}'\n")
print("=== CONTEXT TEXT PACKAGE OUTPUT ===\n")
print(context_package_demo["context_text"])
print("\n" + "="*80 + "\n=== SELECTED DATAFRAME BLOCK ===")
context_package_demo["selected_chunks_df"][["title", "score", "chunk_text"]]

# %%
def build_strict_prompt(query, context_text):
    return f"""You are a grounded RAG assistant specialized in veterinary nutrition.

Rules:
1. Use ONLY the provided context to answer the question. Never add any background knowledge or external facts.
2. If the answer cannot be fully found within the provided context, reply exactly with: "The provided sources do not contain enough information to answer this question."
3. Do not assume or extrapolate. If the context is ambiguous, state only what is explicitly written.
4. Output exactly two sections in this format:
   Answer: [Your grounded answer here]
   Sources: [List the source numbers, titles, and categories used, e.g., Source 1: Title (Category)]

Question: {query}

Context:
{context_text}"""
# %%
target_query_id = "q10"
context_package_demo = build_context_package(target_query_id, retriever="hybrid", k=6)

# Extract components for the prompt
query_text = context_package_demo["query_row"]["query"]
context_text = context_package_demo["context_text"]

# Generate the prompt
final_llm_prompt = build_strict_prompt(query_text, context_text)

print(f"=== GENERATED LLM PROMPT FOR {target_query_id} ===\n")
print(final_llm_prompt)
# %%
import os
import requests

# Changed 0.0.0.0 to 127.0.0.1 (localhost) which is required for making outbound requests on Windows
os.environ["OLLAMA_MODEL"] = "llama3"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:8088"

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

print(f"OLLAMA_HOST={OLLAMA_HOST}")
print(f"OLLAMA_MODEL={OLLAMA_MODEL}")

try:
    tags_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    tags_response.raise_for_status()
    available_models = [model["name"] for model in tags_response.json().get("models", [])]
    print("Available Ollama models:", available_models if available_models else "No local models reported")
    
    if OLLAMA_MODEL not in available_models and f"{OLLAMA_MODEL}:latest" not in available_models:
        print(f"Selected model '{OLLAMA_MODEL}' is not listed. Please pull it first using terminal: ollama pull {OLLAMA_MODEL}")
except Exception as exc:
    print("Could not reach the Ollama server. Start Ollama first, then rerun this cell.")
    print("Connection error:", exc)
# %%
def generate_llama_response(prompt):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0  # Kept at 0.0 for strict adherence to context
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error communicating with Llama: {str(e)}"

# Setup pipeline from previous definitions
target_query_id = "q10"
context_package = build_context_package(target_query_id, retriever="hybrid", k=6)

query_text = context_package["query_row"]["query"]
context_text = context_package["context_text"]

# Build prompt using your custom strict logic
final_prompt = build_strict_prompt(query_text, context_text)

print(f"=== SENDING TO LLAMA ({OLLAMA_MODEL}) ===")
print(f"Query: {query_text}\n")

# Execute generation
llama_answer = generate_llama_response(final_prompt)
print("=== LLAMA RESPONSE ===")
print(llama_answer)
# %%
