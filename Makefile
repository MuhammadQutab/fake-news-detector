setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train_baseline:
	python -m src.models.train_baseline --csv data/dataset.csv --out models/baseline.joblib

train_distilbert:
	python -m src.models.train_distilbert --csv data/dataset.csv --epochs 2 --out models/distilbert

serve:
	streamlit run src/app/streamlit_app.py
