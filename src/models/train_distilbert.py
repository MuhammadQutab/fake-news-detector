import argparse, os, numpy as np
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate

def main(args):
    df = pd.read_csv(args.csv)
    if not {'text','label'}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    label_names = ["real","fake"]
    class_label = ClassLabel(num_classes=2, names=label_names)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(train_df[['text','label']]).class_encode_column("label")
    test_ds  = Dataset.from_pandas(test_df[['text','label']]).class_encode_column("label")

    train_ds = train_ds.map(tok, batched=True)
    test_ds  = test_ds.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved DistilBERT model to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="models/distilbert")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    main(args)
