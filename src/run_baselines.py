import json
import csv
from pathlib import Path

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# This function reads a json file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# This function makes one text string from the dialogue
# We keep the speaker name because it may help the model
def build_input_text(sample):
    turns = sample.get("dialogue", [])
    parts = []

    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


# This function prepares X and y from the dataset
def prepare_data(data):
    texts = []
    labels = []

    for sample in data:
        text = build_input_text(sample)
        label = sample["label"]

        texts.append(text)
        labels.append(label)

    return texts, labels


# This function saves a text file
def save_text(file_path, text):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


# This function saves json
def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# This function saves confusion matrix as csv
def save_confusion_matrix_csv(file_path, cm, labels):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        header = ["true/pred"] + [str(label) for label in labels]
        writer.writerow(header)

        for i, row in enumerate(cm):
            writer.writerow([str(labels[i])] + list(row))


# This function trains one model and saves its results
def train_and_evaluate(model_name, pipeline, x_train, x_val, y_train, y_val, output_dir):
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    # Train the model
    pipeline.fit(x_train, y_train)

    # Predict on validation set
    y_pred = pipeline.predict(x_val)

    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")

    # Make model output folder
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    save_json(model_dir / "metrics.json", metrics)

    # Save classification report
    report_text = classification_report(y_val, y_pred, digits=4, zero_division=0)
    save_text(model_dir / "classification_report.txt", report_text)

    # Save classification report in json form too
    report_json = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    save_json(model_dir / "classification_report.json", report_json)

    # Save validation predictions
    with open(model_dir / "val_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "pred_label"])

        for text, true_label, pred_label in zip(x_val, y_val, y_pred):
            writer.writerow([text, true_label, pred_label])

    # Save confusion matrix
    all_labels = sorted(list(set(y_train + y_val)))
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    save_confusion_matrix_csv(model_dir / "confusion_matrix.csv", cm, all_labels)

    # Save trained pipeline
    dump(pipeline, model_dir / "pipeline.joblib")

    return metrics, pipeline


def main():
    # Project root = one folder above src
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "input_data" / "train.json"
    output_dir = project_root / "outputs"

    # Make outputs folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train data
    train_data = load_json(train_path)

    print(f"Loaded {len(train_data)} training samples.")

    # Prepare text and labels
    texts, labels = prepare_data(train_data)

    # Train/validation split
    # Stratify keeps label balance similar in both splits
    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Training samples   : {len(x_train)}")
    print(f"Validation samples : {len(x_val)}")

    # Solution 1
    # TF-IDF + Logistic Regression
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    # Solution 2
    # TF-IDF + Linear SVM
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
        )),
    ])

    # Run both solutions
    lr_metrics, lr_model = train_and_evaluate(
        model_name="tfidf_logistic_regression",
        pipeline=lr_pipeline,
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        output_dir=output_dir,
    )

    svm_metrics, svm_model = train_and_evaluate(
        model_name="tfidf_linear_svm",
        pipeline=svm_pipeline,
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        output_dir=output_dir,
    )

    # Save one summary file for easy report writing
    summary_rows = [lr_metrics, svm_metrics]

    with open(output_dir / "results_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "macro_f1", "weighted_f1"])

        for row in summary_rows:
            writer.writerow([
                row["model"],
                f'{row["accuracy"]:.4f}',
                f'{row["macro_f1"]:.4f}',
                f'{row["weighted_f1"]:.4f}',
            ])

    # Choose best model using macro F1
    if lr_metrics["macro_f1"] >= svm_metrics["macro_f1"]:
        best_name = "tfidf_logistic_regression"
        best_model = lr_model
        best_metrics = lr_metrics
    else:
        best_name = "tfidf_linear_svm"
        best_model = svm_model
        best_metrics = svm_metrics

    dump(best_model, output_dir / "best_pipeline.joblib")
    save_text(output_dir / "best_model.txt", best_name)
    save_json(output_dir / "best_metrics.json", best_metrics)

    print(f"\n{'=' * 60}")
    print("Done")
    print(f"{'=' * 60}")
    print("Results saved in:", output_dir)
    print("Best model:", best_name)


if __name__ == "__main__":
    main()