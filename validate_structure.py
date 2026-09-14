from pathlib import Path

root = Path(__file__).resolve().parent
required = [
    "README.md", "requirements.txt", "data_pipeline/pipeline.py",
    "analytics/analysis.py", "support_assistant/main.py",
    "support_assistant/ingest.py", "support_assistant/Dockerfile",
]
missing = [path for path in required if not (root / path).exists()]
docs = list((root / "support_assistant/docs").glob("doc_*.txt"))
if missing or len(docs) != 8:
    raise SystemExit(f"Validation failed. Missing={missing}, documents={len(docs)}")
print("Structure validation passed: all modules and 8 policy documents are present.")
