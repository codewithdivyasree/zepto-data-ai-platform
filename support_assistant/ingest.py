from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).resolve().parent
COLLECTION = "zepto_policies"


# STEP 1: Open the local ChromaDB collection.
def get_collection():
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(
        COLLECTION, embedding_function=embedding, metadata={"hnsw:space": "cosine"}
    )


# STEP 2: Read and store all eight policy documents.
def ingest():
    collection = get_collection()
    paths = sorted((ROOT / "docs").glob("doc_*.txt"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(paths)}")
    ids = []
    documents = []
    metadata = []

    for path in paths:
        document_id = path.stem
        document_text = path.read_text(encoding="utf-8").strip()

        ids.append(document_id)
        documents.append(document_text)
        metadata.append({"source": document_id})

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadata,
    )
    print(f"Indexed {collection.count()} policy documents")
    return collection


if __name__ == "__main__":
    ingest()
