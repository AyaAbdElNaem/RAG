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

from dotenv import load_dotenv
load_dotenv()

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
# NOTE: This dataset is used EXCLUSIVELY by the evaluation section further
# below (precision/recall/hit-rate/MRR dashboard). It is never consulted by
# the /chat endpoint anymore. Kept 100% unchanged per requirements.
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
# 4b. EVALUATION METRICS (GROUND-TRUTH BASED) — UNCHANGED, EVAL-ONLY
# ==============================================================================
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
# Performance Evaluation Dashboard & Live Sample Target Matching
# (Ground truth is used HERE ONLY — never inside the /chat endpoint)
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
# 5. CONTEXT BUILDING MODULE (REFACTORED — QUERY-DRIVEN, NO GROUND TRUTH)
# ==============================================================================
# BEFORE: build_context_package(query_id) looked up the question text from
#         queries_df using a ground-truth query_id.
# AFTER:  build_context_package(query, ...) accepts the *real user question*
#         directly and retrieves against it. queries_df / ground_truth are no
#         longer touched anywhere in this function.
def build_context_package(query: str, retriever="hybrid", k=6, word_budget=300, max_chunks=3):
    query_text = (query or "").strip()

    if not query_text:
        raise ValueError("A non-empty query string is required.")

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
        "query": query_text,
        "candidates": candidates,
        "selected_chunks_df": selected_df,
        "used_words": used_words,
        "context_text": "\n\n".join(context_blocks),
    }

# Demo run using a plain question (no query_id / ground truth involved)
demo_question = "What are the three most common food allergens in domestic dogs?"
context_package_demo = build_context_package(demo_question, retriever="hybrid", k=6)

print(f"=== CONTEXT GENERATION FOR DEMO QUESTION ===")
print(f"Testing Query: '{context_package_demo['query']}'\n")
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
# Demo run of the full (non-ground-truth) pipeline for a sample question
demo_question = "What is the best feeding regimen to adult dogs?"
context_package_demo = build_context_package(demo_question, retriever="hybrid", k=6)

query_text = context_package_demo["query"]
context_text = context_package_demo["context_text"]

final_llm_prompt = build_strict_prompt(query_text, context_text)

print(f"=== GENERATED LLM PROMPT ===\n")
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
# NOTE: this function was left commented-out (dead code) in the original
# notebook, which meant /chat could never actually generate an answer.
# Re-enabled as-is (logic untouched) so the chatbot endpoint below works.
import os
from groq import Groq

def generate_llama_response(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error communicating with Llama: GROQ_API_KEY environment variable is not set"

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Llama: {str(e)}"

# Demo run of the full pipeline against a plain user question
demo_question = "What is the best feeding regimen to adult dogs?"
context_package = build_context_package(demo_question, retriever="hybrid", k=6)

query_text = context_package["query"]
context_text = context_package["context_text"]

final_prompt = build_strict_prompt(query_text, context_text)

print(f"=== SENDING TO LLAMA (llama-3.1-8b-instant) ===")
print(f"Query: {query_text}\n")

llama_answer = generate_llama_response(final_prompt)
print("=== LLAMA RESPONSE ===")
print(llama_answer)
# %%
# ==========================================
# STEP 4: CONVERT TO INTERACTIVE API SERVER
# ==========================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# ==============================================================================
# GREETING / PLATFORM-QUESTION HANDLING (NEW)
# ==============================================================================
# Casual greetings ("hi", "hello", "hey"...) or general questions about the
# platform itself ("who are you?", "what is this app?") should never be routed
# through the RAG pipeline. Retrieval against a greeting returns irrelevant
# chunks and the strict prompt forces the model to reply with the unhelpful
# "the provided sources do not contain enough information" line. Instead we
# detect these up front and answer directly with a warm introduction, before
# build_context_package / generate_llama_response are ever called.

WELCOME_MESSAGE = (
    "Hi there! 👋 I'm **PetNutri 🐾**, an AI platform dedicated to helping you "
    "understand your pet's nutritional and dietary needs. "
    "Ask me anything about ingredients, feeding schedules, life-stage nutrition, "
    "or diet plans for your dog or cat, and I'll answer using our veterinary "
    "nutrition sources!"
)

# Short, standalone greetings matched as a whole message (after stripping
# punctuation), so a real question like "hi, what should I feed my puppy?"
# still goes through RAG normally.
_GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "hey", "heya", "hiya", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "salam", "assalamualaikum", "howdy", "sup", "what's up", "whats up",
}

# Substrings that indicate a question about the platform/assistant itself
# rather than a nutrition question.
_PLATFORM_PHRASES = (
    "who are you", "what are you", "what is petnutri", "what's petnutri",
    "what is this app", "what is this platform", "what can you do",
    "what do you do", "how do you work", "tell me about yourself",
    "tell me about petnutri", "introduce yourself", "who made you",
    "what is this website", "help me get started",
)


def is_greeting_or_platform_question(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s']", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    if not normalized:
        return False

    if normalized in _GREETING_WORDS:
        return True

    # Very short messages (<= 3 words) that are just a greeting word plus
    # filler ("hi there", "hey!!") should also count.
    words = normalized.split()
    if len(words) <= 3 and any(word in _GREETING_WORDS for word in words):
        return True

    if any(phrase in normalized for phrase in _PLATFORM_PHRASES):
        return True

    return False


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Real chatbot endpoint.

    Flow:
        user_question -> retrieve_top_k_hybrid -> context
                       -> build_strict_prompt -> generate_llama_response -> answer

    Ground truth / query_id / queries_df are NEVER used here.

    Casual greetings and general "what is this platform" questions are
    intercepted before retrieval and answered directly (see
    is_greeting_or_platform_question above).
    """
    try:
        user_question = (request.question or "").strip()

        if not user_question:
            return {"reply": "Please type a question about your pet's nutrition.", "sources": []}

        if is_greeting_or_platform_question(user_question):
            return {"reply": WELCOME_MESSAGE, "sources": []}

        context_package = build_context_package(
            query=user_question,
            retriever="hybrid",
            k=6,
        )

        final_prompt = build_strict_prompt(user_question, context_package["context_text"])

        llama_answer = generate_llama_response(final_prompt)

        sources_df = context_package["selected_chunks_df"]
        sources = (
            sources_df[["title", "category", "source", "score"]].to_dict("records")
            if not sources_df.empty
            else []
        )

        return {
            "reply": llama_answer,
            "sources": sources,
        }

    except Exception as e:
        return {"reply": f"Error in RAG Pipeline: {str(e)}", "sources": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

