# Expected Output Generator

Streamlit app to generate and refine `expected_output` values for dataset rows using OpenAI models.

## Quickstart
```bash
cd expected_output_generator
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Set `OPENAI_API_KEY` in your environment before running.

## Workflow
- Upload CSV or Excel with an `input` column (adds `expected_output` and `feedback` if missing).
- Configure model, base prompt, and JSON schema instructions in the sidebar.
- Work row by row, add feedback, and regenerate.
- Use “Generate for All Empty Rows” to fill blanks sequentially.
- Download updated sheet as CSV or Excel.
