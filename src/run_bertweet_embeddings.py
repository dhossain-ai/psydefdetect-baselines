import json
import csv
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


MODEL_NAME = "vinai/bertweet-base"
BATCH_SIZE = 16
MAX_LENGTH = 128
RANDOM_STATE = 42


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(file_path, text):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_confusion_matrix_csv(file_path, cm, labels):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + [str(label) for label in labels])

        for i, row in enumerate(cm):
            writer.writerow([str(labels[i])] + list(row))


def dialogue_to_text(sample):
    parts = []

    for turn in sample.get("dialogue", []):
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


def last_n_turns_to_text(sample, n=3):
    turns = sample.get("dialogue", [])[-n:]
    parts = []

    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


def build_input_text(sample, input_mode):
    current_text = sample.get("current_text", "").strip()
    full_dialogue = dialogue_to_text(sample)
    last3 = last_n_turns_to_text(sample, n=3)

    if input_mode == "current_text":
        return current_text

    if input_mode == "current_plus_last3":
        return f"CURRENT: {current_text} CONTEXT: {last3}"

    if input_mode == "current_plus_dialogue":
        return f"CURRENT: {current_text} CONTEXT: {full_dialogue}"

    if input_mode == "full_dialogue":
        return full_dialogue

    raise ValueError(f"Unknown input_mode: {input_mode}")


def prepare_data(data, input_mode):
    texts = []
    labels = []

    for sample in data:
        texts.append(build_input_text(sample, input_mode))
        labels.append(sample["label"])

    return texts, labels


def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling over non-padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def extract_embeddings(texts, tokenizer, model, device):
    """
    Convert text examples into fixed-size BERTweet embeddings.
    """
    model.eval()
    all_embeddings = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting embeddings"):
        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def train_and_evaluate_classifier(
    experiment_name,
    classifier_pipeline,
    x_train_embeddings,
    x_val_embeddings,
    y_train,
    y_val,
    output_dir,
):
    print(f"\n{'=' * 80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 80}")

    classifier_pipeline.fit(x_train_embeddings, y_train)
    y_pred = classifier_pipeline.predict(x_val_embeddings)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": experiment_name,
        "model_name": MODEL_NAME,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    save_json(experiment_dir / "metrics.json", metrics)

    report_text = classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0,
    )
    save_text(experiment_dir / "classification_report.txt", report_text)

    report_json = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    save_json(experiment_dir / "classification_report.json", report_json)

    with open(experiment_dir / "val_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label"])

        for true_label, pred_label in zip(y_val, y_pred):
            writer.writerow([true_label, pred_label])

    all_labels = sorted(list(set(y_train + y_val)))
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    save_confusion_matrix_csv(experiment_dir / "confusion_matrix.csv", cm, all_labels)

    dump(classifier_pipeline, experiment_dir / "classifier.joblib")

    return metrics, classifier_pipeline


def main():
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "input_data" / "train.json"

    output_dir = project_root / "outputs_final" / "bertweet_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        print("Error: train.json not found at:", train_path)
        return

    train_data = load_json(train_path)
    print(f"Loaded training samples: {len(train_data)}")

    labels = [sample["label"] for sample in train_data]
    indices = list(range(len(train_data)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_samples = [train_data[i] for i in train_idx]
    val_samples = [train_data[i] for i in val_idx]

    print(f"Training samples   : {len(train_samples)}")
    print(f"Validation samples : {len(val_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading tokenizer and model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    experiments = [
        {
            "input_mode": "current_text",
            "name_prefix": "bertweet_current_text",
            "description": "BERTweet embeddings using current utterance only",
        },
        {
            "input_mode": "current_plus_last3",
            "name_prefix": "bertweet_current_plus_last3",
            "description": "BERTweet embeddings using current utterance plus last 3 turns",
        },
    ]

    all_results = []
    best_metrics = None
    best_classifier = None
    best_experiment_name = None
    best_input_mode = None

    for exp in experiments:
        print(f"\nPreparing input mode: {exp['input_mode']}")

        x_train_texts, y_train = prepare_data(train_samples, exp["input_mode"])
        x_val_texts, y_val = prepare_data(val_samples, exp["input_mode"])

        x_train_embeddings = extract_embeddings(
            x_train_texts,
            tokenizer,
            model,
            device,
        )

        x_val_embeddings = extract_embeddings(
            x_val_texts,
            tokenizer,
            model,
            device,
        )

        # Save embeddings, useful if you want to rerun only classifiers later
        np.save(output_dir / f"{exp['name_prefix']}_x_train_embeddings.npy", x_train_embeddings)
        np.save(output_dir / f"{exp['name_prefix']}_x_val_embeddings.npy", x_val_embeddings)
        np.save(output_dir / f"{exp['name_prefix']}_y_train.npy", np.array(y_train))
        np.save(output_dir / f"{exp['name_prefix']}_y_val.npy", np.array(y_val))

        classifiers = [
            {
                "suffix": "logreg",
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    )),
                ]),
            },
            {
                "suffix": "svm",
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LinearSVC(
                        class_weight="balanced",
                        C=1.0,
                        max_iter=10000,
                    )),
                ]),
            },
        ]

        for clf_exp in classifiers:
            experiment_name = f"{exp['name_prefix']}_{clf_exp['suffix']}"

            metrics, trained_classifier = train_and_evaluate_classifier(
                experiment_name=experiment_name,
                classifier_pipeline=clf_exp["pipeline"],
                x_train_embeddings=x_train_embeddings,
                x_val_embeddings=x_val_embeddings,
                y_train=y_train,
                y_val=y_val,
                output_dir=output_dir,
            )

            metrics["input_mode"] = exp["input_mode"]
            metrics["classifier"] = clf_exp["suffix"]
            metrics["description"] = exp["description"]

            all_results.append(metrics)

            if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_metrics = metrics
                best_classifier = trained_classifier
                best_experiment_name = experiment_name
                best_input_mode = exp["input_mode"]

    summary_path = output_dir / "bertweet_results_summary.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "input_mode",
            "classifier",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "description",
        ])

        for row in all_results:
            writer.writerow([
                row["experiment"],
                row["input_mode"],
                row["classifier"],
                f'{row["accuracy"]:.4f}',
                f'{row["macro_f1"]:.4f}',
                f'{row["weighted_f1"]:.4f}',
                row["description"],
            ])

    dump(best_classifier, output_dir / "best_bertweet_classifier.joblib")
    save_text(output_dir / "best_bertweet_model.txt", best_experiment_name)
    save_text(output_dir / "best_bertweet_input_mode.txt", best_input_mode)
    save_json(output_dir / "best_bertweet_metrics.json", best_metrics)

    print(f"\n{'=' * 80}")
    print("BERTweet embedding experiments completed")
    print(f"{'=' * 80}")
    print("Results saved in:", output_dir)
    print("Summary:", summary_path)
    print("Best BERTweet experiment:", best_experiment_name)
    print(f"Best macro F1: {best_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()