import math

from foundry_local_sdk import Configuration, FoundryLocalManager
from db import get_all_documents

FoundryLocalManager.initialize(Configuration(app_name="rag-assistant"))
manager = FoundryLocalManager.instance

embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")

if not embedding_model.is_cached:
    print("Downloading embedding model...")
    embedding_model.download(lambda p: print(f"\r%{p:.0f}", end="", flush=True))
    print()

embedding_model.load()

embedding_client = embedding_model.get_embedding_client()


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def get_top_chunks(query, k=2):
    """Return the top-k most relevant document chunks for a given query."""
    documents = get_all_documents()
    if not documents:
        return []

    query_result = embedding_client.generate_embeddings([query])
    query_vector = query_result.data[0].embedding

    scored = []
    for doc in documents:
        score = cosine_similarity(query_vector, doc["embedding"])
        scored.append((score, doc["content"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    test_query = "Is free will an illusion?"
    top_chunks = get_top_chunks(test_query, k=9)

    print(f"Query: {test_query}\n")
    print("All chunks ranked by similarity:")
    for score, content in top_chunks:
        print(f"  {score:.4f}  {content[:60]!r}")

    embedding_model.unload()