# Fake News Detection — Baseline TF‑IDF → DistilBERT

A clean, reproducible fake-news classifier built with free tools. Start with a lightweight TF‑IDF + Logistic Regression baseline, then upgrade to a fine‑tuned DistilBERT. Includes a Streamlit demo app.

## Demo (local)

```bash
# create venv and install deps
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train baseline (uses data/dataset.csv)
python -m src.models.train_baseline --csv data/dataset.csv --out models/baseline.joblib

# 2) Serve Streamlit app (uses whichever model exists; prefers DistilBERT if present)
streamlit run src/app/streamlit_app.py
```


```
## 🧠 Model Weights (Download & Place) 

This app expects the trained model files to be present locally.

1. **Download the weights** from the latest release:  
   ➡️ **[fake_real_model_v1.zip] ([ (https://github.com/MuhammadQutab/fake-news-detector/releases/download/v1.1.0/fake_real_model_v1.zip) ])**

2. **Unzip** into this exact folder:
models/fake_real_model/

sql
Copy code
Your folder should then look like:
models/fake_real_model/
├─ config.json
├─ pytorch_model.bin (or model.safetensors)
├─ tokenizer.json
├─ tokenizer_config.json
├─ vocab.txt
└─ special_tokens_map.json
```


## DistilBERT fine‑tune (optional but recommended)

```bash
# This uses Hugging Face Transformers + Datasets (all free).
python -m src.models.train_distilbert --csv data/dataset.csv --epochs 2 --out models/distilbert
```


## LinkedIn Video Link:


https://www.linkedin.com/posts/muhammadqutab03_python-nlp-transformers-activity-7384967969140891648-dyit?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEDX9coBoMdRTqxs5ojMW8ScnBWaDJMcSkc


## Expected CSV format
`data/dataset.csv` with columns: `text,label` where label is 0 (real) or 1 (fake). A tiny sample is included so you can test the pipeline.

## Results to report
- Baseline: Precision/Recall/F1 (macro), Confusion Matrix
- DistilBERT: same metrics + validation loss plot
- Error analysis: short examples of false positives/negatives

## Project structure
```
fake-news-detector/
├─ data/                # your CSV here
├─ models/              # saved models
├─ reports/             # figures
├─ src/
│  ├─ data/prepare.py
│  ├─ eval/metrics.py
│  ├─ models/train_baseline.py
│  ├─ models/train_distilbert.py
│  └─ app/streamlit_app.py
└─ requirements.txt
```

## License
MIT
