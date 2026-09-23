# Support Assistant

This module is organized as a simple four-stage RAG flow:

1. **Ingestion:** `ingest.py` reads the eight files in `docs/`.
2. **Embedding:** `all-MiniLM-L6-v2` creates local embeddings stored in ChromaDB.
3. **Retrieval:** `retrieve_and_answer` retrieves the three closest policy documents.
4. **Generation:** mock mode returns a deterministic snippet from the best document.

## Run locally

From the repository root:

```bash
pip install -r support_assistant/requirements.txt
python support_assistant/ingest.py
```

Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860/docs` to test `POST /ask` using FastAPI's interface.

Policy example:

```json
{"query": "What is the delivery fee?"}
```

Raw mock-mode response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_02"],
  "confidence": 1.0
}
```

General example:

```json
{"query": "Who invented Python?"}
```

Raw mock-mode response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Both responses follow the validated schema `answer`, `sources`, and `confidence`. `MOCK_LLM=1` is the default graded path and makes no LLM API call.
