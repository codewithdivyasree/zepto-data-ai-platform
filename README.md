# Zepto Data & AI Platform

A single capstone repository containing three connected modules:

1. `data_pipeline` — scrape, clean, convert, normalize, store and query catalogue data.
2. `analytics` — profile Titanic data, perform EDA, train/evaluate models and save a deployable pipeline.
3. `support_assistant` — an offline-first RAG policy assistant using ChromaDB, LangGraph and FastAPI.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Module 1

```bash
python data_pipeline/pipeline.py
```

The script scrapes at least 60 books from at least three categories, cleans the fields, converts GBP to INR using the fixed project baseline **1 GBP = 105.50 INR**, creates a normalized SQLite database, runs the required SQL queries, and compares SQL `JOIN` output with `pandas.merge` output.

Cleaning decisions: invalid numeric fields are converted to missing values and median-imputed when possible. Rows with an invalid rating or missing essential identifiers are dropped because inventing a book title, category, or rating would make the catalogue misleading. Availability is parsed into a Boolean value.

## Run Module 2

```bash
python analytics/analysis.py
```

On its first run, the script loads Titanic once with Seaborn and immediately saves `analytics/titanic.csv`. Later runs use that committed CSV. It generates EDA results/charts, trains three classifiers on one stratified split, evaluates imbalance strategies, tunes Random Forest, runs the regression side-task, and saves the best complete pipeline.

## Run Module 3

```bash
python support_assistant/ingest.py
MOCK_LLM=1 uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

Example retrieval request:

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the delivery fee?"}'
```

Example general request:

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query":"Who invented Python?"}'
```

Responses always follow:

```json
{"answer":"...","sources":["doc_01"],"confidence":1.0}
```

### RAG architecture

- **Ingestion:** `ingest.py` loads the eight files from `support_assistant/docs` and treats each short policy document as one chunk.
- **Embedding:** `all-MiniLM-L6-v2` creates local embeddings; the `zepto_policies` ChromaDB collection stores vectors, text and document IDs.
- **Retrieval:** the LangGraph `retrieve_and_answer` node embeds a policy question and retrieves the three nearest chunks with cosine-distance search.
- **Generation:** in the default `MOCK_LLM=1` mode, deterministic code returns a snippet from the highest-ranked chunk. `direct_answer` returns a fixed scope message for general questions. With optional `MOCK_LLM=0`, only classification and generation change to a configured real-LLM implementation; local embedding and retrieval remain the same.

### Structured prompt design

The optional real-LLM branch uses the role–context–task–format–length prompt in `support_assistant/prompts.py`. It includes a grounding constraint, a negative constraint and a few-shot example. The default graded mock path does not call an LLM.

## Docker

```bash
docker build -t zepto-support ./support_assistant
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

## Git workflow evidence

Before submission, create a feature branch, make at least two meaningful commits on it, and merge it into `main` with a merge commit. Git history cannot be fabricated by source code, so perform this workflow while developing or committing this project.


