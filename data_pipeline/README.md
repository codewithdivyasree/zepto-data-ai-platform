# Data Pipeline

Run `python pipeline.py` from this directory or `python data_pipeline/pipeline.py` from the repository root. The code is divided into seven simple numbered steps. The generated artifacts are `books.csv`, `books.db`, and `query_outputs.txt`.

The mandatory conversion uses the fixed assignment rate **1 GBP = 105.50 INR**. Invalid prices are coerced to missing and median-imputed; rows missing essential categorical values are dropped and the decision is logged.
