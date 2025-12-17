# Evals Generator (Streamlit)

Lightweight Streamlit app to draft evaluation items and export them to JSONL.

## Features
- Add/edit/delete eval items with input, expected output, and optional metadata.
- Import items from CSV or JSONL.
- Export current items as JSONL for downstream eval pipelines.

## Quickstart
```bash
cd evals_generator
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Usage
1. Open the app and add items via the form (input + expected output are required; metadata optional JSON).
2. Import CSV/JSONL if you already have items.
3. Download JSONL when ready to use in your eval runner.
