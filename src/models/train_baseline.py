import argparse, joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data.prepare import load_and_split
from src.eval.metrics import print_report, plot_confusion

def main(args):
    X_train, X_test, y_train, y_test = load_and_split(args.csv, test_size=0.2, seed=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300, n_jobs=None))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print_report(y_test, y_pred)
    os.makedirs('reports', exist_ok=True)
    plot_confusion(y_test, y_pred, out_path='reports/confusion_baseline.png')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Saved baseline model to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="models/baseline.joblib")
    args = p.parse_args()
    main(args)
