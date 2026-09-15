# Analytics Pipeline

Run `python analysis.py`. The file is organized into simple numbered steps with beginner-friendly comments. It loads Titanic from Seaborn only when the offline `titanic.csv` fallback does not yet exist, then uses the same DataFrame for all subsequent work.

Generated written evidence is saved in `analysis_report.md`; charts go to `charts/`; the best complete preprocessing-and-model pipeline is saved as `best_classifier_pipeline.joblib` and reloaded for a raw-input prediction check.
