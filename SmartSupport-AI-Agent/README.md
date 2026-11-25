# SmartSupport — Customer Support AI Agent

Runnable portfolio demo with an intent classifier, TF-IDF knowledge retrieval, SQLite ticket tool, FastAPI backend, and Streamlit chat UI. Includes an optional Hugging Face + PEFT LoRA training script.

## Windows setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.api:app --reload --port 8001
```
Second terminal:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```
API docs: http://127.0.0.1:8001/docs

Test: `pytest -q`. The included CSV is a tiny demo dataset; expand it and evaluate on a held-out dataset before reporting model metrics.

## Optional LoRA demo
Install `requirements-training.txt`, then run:
`python training/train_lora.py --model distilgpt2 --output_dir artifacts/lora-support`
Requires internet for model download and suitable memory. This is a tiny learning demo, not a production-trained support model.

The agent only creates demo tickets, checks ticket status, and searches a local help file. Do not connect real customer systems without authentication, authorization, audit logging, and human approval.
