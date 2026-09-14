from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).resolve().parent
COLLECTION = "zepto_policies"


def get_collection():
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(
        COLLECTION, embedding_function=embedding, metadata={"hnsw:space": "cosine"}
    )


def ingest():
    collection = get_collection()
    paths = sorted((ROOT / "docs").glob("doc_*.txt"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(paths)}")
    ids = [p.stem for p in paths]
    documents = [p.read_text(encoding="utf-8").strip() for p in paths]
    collection.upsert(ids=ids, documents=documents, metadatas=[{"source": i} for i in ids])
    print(f"Indexed {collection.count()} policy documents")
    return collection


if __name__ == "__main__":
    ingest()

