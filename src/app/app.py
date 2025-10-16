# src/app/app.py
from pathlib import Path
import io
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# ---------- Resolve model directory (do NOT show path in UI) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # fake-news-detector/
CANDIDATES = [
    PROJECT_ROOT / "models" / "fake_real_model",
    PROJECT_ROOT / "fake_real_model",
]

MODEL_DIR = None
for p in CANDIDATES:
    if (p / "config.json").exists():
        MODEL_DIR = p
        break
    if (p / "fake_real_model" / "config.json").exists():  # guard double-nesting
        MODEL_DIR = p / "fake_real_model"
        break

if MODEL_DIR is None:
    st.error(
        "Model not found.\n\n"
        "Expected: models/fake_real_model/config.json under your project root.\n"
        "Fix:\n"
        "  1) Unzip the model.\n"
        "  2) Ensure config.json is directly inside models/fake_real_model/ (no extra folder level).\n"
        "  3) Run from project root:  streamlit run src/app/app.py"
    )
    st.stop()

# ---------- Load model/tokenizer ----------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    st.error("Failed to load model. See details below.")
    st.exception(e)
    st.stop()

LABELS = ["Fake", "Real"]
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _infer_column_options(columns):
    # suggest likely text columns
    lower = [c.lower() for c in columns]
    for cand in ["text", "headline", "title", "content", "statement"]:
        if cand in lower:
            return columns[lower.index(cand)]
    return columns[0] if columns else None

def predict_texts(texts):
    enc = tokenizer(
        list(texts), return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs

# ---------- UI ----------
st.title("📰 Fake News Detection")
tabs = st.tabs(["Single Text", "CSV Upload"])

# ---- Tab 1: Single text ----
with tabs[0]:
    st.caption("Enter a news headline or short statement to classify.")
    text = st.text_area("Text", height=140, key="single_text")
    if st.button("Predict", key="single_btn"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            preds, probs = predict_texts([text])
            pred = int(preds[0])
            st.success(f"Prediction: **{LABELS[pred]}**  |  Confidence: {probs[0][pred]:.2f}")
            with st.expander("Scores"):
                st.json({"conf_fake": float(probs[0][0]), "conf_real": float(probs[0][1])})

# ---- Tab 2: CSV upload ----
with tabs[1]:
    st.caption("Upload a CSV. The app will add predictions and let you download the results.")
    file = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    if file:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(io.BytesIO(file.getvalue()))  # fallback for some locales

        if df.empty:
            st.error("The CSV seems empty.")
        else:
            st.write("Preview:", df.head(10))
            text_col_guess = _infer_column_options(df.columns.tolist())
            text_col = st.selectbox(
                "Select the column that contains the text to classify:",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(text_col_guess) if text_col_guess in df.columns else 0,
            )

            batch_size = st.slider("Batch size", 16, 256, 64, step=16)
            if st.button("Run batch predictions"):
                texts = df[text_col].astype(str).tolist()

                preds_all, conf_fake, conf_real = [], [], []
                for i in st.progress(0, text="Running inference..."), range(0, len(texts), batch_size):
                    pass  # dummy to reserve variable

                # proper loop with progress
                prog = st.progress(0, text="Running inference...")
                total = len(texts)
                for i in range(0, total, batch_size):
                    batch = texts[i : i + batch_size]
                    preds, probs = predict_texts(batch)
                    preds_all.extend(preds.tolist())
                    conf_fake.extend(probs[:, 0].tolist())
                    conf_real.extend(probs[:, 1].tolist())
                    prog.progress(min(i + batch_size, total) / total)

                out = df.copy()
                out["prediction"] = [LABELS[p] for p in preds_all]
                out["conf_fake"] = conf_fake
                out["conf_real"] = conf_real
                out["confidence"] = out[["conf_fake", "conf_real"]].max(axis=1)

                st.success("Done! Preview of results:")
                st.dataframe(out.head(20), use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )