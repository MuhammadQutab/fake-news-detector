import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(csv_path, test_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    if not {'text','label'}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=seed, stratify=df['label']
    )
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()
