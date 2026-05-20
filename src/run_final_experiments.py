import json
import csv
from pathlib import Path

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


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
    """
    Convert full dialogue into one string.
    Same idea as Milestone 1: keep speaker names.
    """
    parts = []

    for turn in sample.get("dialogue", []):
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


def last_n_turns_to_text(sample, n=3):
    """
    Convert only the last n dialogue turns into text.
    This reduces noise compared with full dialogue.
    """
    turns = sample.get("dialogue", [])[-n:]
    parts = []

    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


def build_input_text(sample, input_mode):
    """
    Different input variants for final milestone experiments.

    input_mode options:
    - current_text
    - full_dialogue
    - current_plus_dialogue
    - current_plus_last3
    """
    current_text = sample.get("current_text", "").strip()
    full_dialogue = dialogue_to_text(sample)
    last3 = last_n_turns_to_text(sample, n=3)

    if input_mode == "current_text":
        return current_text

    if input_mode == "full_dialogue":
        return full_dialogue

    if input_mode == "current_plus_dialogue":
        return f"CURRENT: {current_text} CONTEXT: {full_dialogue}"

    if input_mode == "current_plus_last3":
        return f"CURRENT: {current_text} CONTEXT: {last3}"

    raise ValueError(f"Unknown input_mode: {input_mode}")


def prepare_data(data, input_mode):
    texts = []
    labels = []

    for sample in data:
        texts.append(build_input_text(sample, input_mode))
        labels.append(sample["label"])

    return texts, labels


def make_pipeline(feature_type, classifier_type):
    """
    Create model pipelines for different n-gram experiments.

    feature_type options:
    - word_1_2
    - word_1_3
    - word_1_4
    - char_3_5
    - word_char

    classifier_type options:
    - logreg
    - svm
    """
    if feature_type == "word_1_2":
        features = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )

    elif feature_type == "word_1_3":
        features = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=50000,
        )

    elif feature_type == "word_1_4":
        features = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 4),
            min_df=2,
            max_features=70000,
        )

    elif feature_type == "char_3_5":
        features = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
        )

    elif feature_type == "word_char":
        word_features = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=50000,
        )

        char_features = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
        )

        features = FeatureUnion([
            ("word_tfidf", word_features),
            ("char_tfidf", char_features),
        ])

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    if classifier_type == "logreg":
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )

    elif classifier_type == "svm":
        classifier = LinearSVC(
            class_weight="balanced",
            C=1.0,
        )

    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    return Pipeline([
        ("features", features),
        ("clf", classifier),
    ])


def train_and_evaluate(
    experiment_name,
    pipeline,
    x_train,
    x_val,
    y_train,
    y_val,
    output_dir,
):
    print(f"\n{'=' * 80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 80}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val)

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
        writer.writerow(["text", "true_label", "pred_label"])

        for text, true_label, pred_label in zip(x_val, y_val, y_pred):
            writer.writerow([text, true_label, pred_label])

    all_labels = sorted(list(set(y_train + y_val)))
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    save_confusion_matrix_csv(experiment_dir / "confusion_matrix.csv", cm, all_labels)

    dump(pipeline, experiment_dir / "pipeline.joblib")

    return metrics, pipeline


def main():
    project_root = Path(__file__).resolve().parent.parent

    train_path = project_root / "input_data" / "train.json"
    output_dir = project_root / "outputs_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        print("Error: train.json not found at:", train_path)
        return

    train_data = load_json(train_path)
    print(f"Loaded training samples: {len(train_data)}")

    labels = [sample["label"] for sample in train_data]

    # Use indices so every experiment uses exactly the same validation split.
    indices = list(range(len(train_data)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_samples = [train_data[i] for i in train_idx]
    val_samples = [train_data[i] for i in val_idx]

    print(f"Training samples   : {len(train_samples)}")
    print(f"Validation samples : {len(val_samples)}")

    experiments = [
        # Repeated Milestone 1 settings for comparison
        {
            "name": "m1_full_dialogue_word_1_2_logreg",
            "input_mode": "full_dialogue",
            "feature_type": "word_1_2",
            "classifier_type": "logreg",
            "description": "Milestone 1 baseline: full dialogue, word TF-IDF 1-2, Logistic Regression",
        },
        {
            "name": "m1_full_dialogue_word_1_2_svm",
            "input_mode": "full_dialogue",
            "feature_type": "word_1_2",
            "classifier_type": "svm",
            "description": "Milestone 1 baseline: full dialogue, word TF-IDF 1-2, Linear SVM",
        },

        # Final milestone: input comparison
        {
            "name": "current_text_word_1_3_svm",
            "input_mode": "current_text",
            "feature_type": "word_1_3",
            "classifier_type": "svm",
            "description": "Current utterance only, word TF-IDF 1-3, Linear SVM",
        },
        {
            "name": "full_dialogue_word_1_3_svm",
            "input_mode": "full_dialogue",
            "feature_type": "word_1_3",
            "classifier_type": "svm",
            "description": "Full dialogue, word TF-IDF 1-3, Linear SVM",
        },
        {
            "name": "current_plus_dialogue_word_1_3_svm",
            "input_mode": "current_plus_dialogue",
            "feature_type": "word_1_3",
            "classifier_type": "svm",
            "description": "Current utterance plus full dialogue, word TF-IDF 1-3, Linear SVM",
        },
        {
            "name": "current_plus_last3_word_1_3_svm",
            "input_mode": "current_plus_last3",
            "feature_type": "word_1_3",
            "classifier_type": "svm",
            "description": "Current utterance plus last 3 turns, word TF-IDF 1-3, Linear SVM",
        },

        # Final milestone: stronger n-gram settings
        {
            "name": "current_plus_last3_word_1_4_svm",
            "input_mode": "current_plus_last3",
            "feature_type": "word_1_4",
            "classifier_type": "svm",
            "description": "Current utterance plus last 3 turns, word TF-IDF 1-4, Linear SVM",
        },
        {
            "name": "current_plus_last3_char_3_5_svm",
            "input_mode": "current_plus_last3",
            "feature_type": "char_3_5",
            "classifier_type": "svm",
            "description": "Current utterance plus last 3 turns, character TF-IDF 3-5, Linear SVM",
        },
        {
            "name": "current_plus_last3_word_char_svm",
            "input_mode": "current_plus_last3",
            "feature_type": "word_char",
            "classifier_type": "svm",
            "description": "Current utterance plus last 3 turns, combined word and character TF-IDF, Linear SVM",
        },
    ]

    all_results = []
    best_metrics = None
    best_pipeline = None
    best_experiment = None

    for exp in experiments:
        input_mode = exp["input_mode"]

        x_train, y_train = prepare_data(train_samples, input_mode)
        x_val, y_val = prepare_data(val_samples, input_mode)

        pipeline = make_pipeline(
            feature_type=exp["feature_type"],
            classifier_type=exp["classifier_type"],
        )

        metrics, trained_pipeline = train_and_evaluate(
            experiment_name=exp["name"],
            pipeline=pipeline,
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            output_dir=output_dir,
        )

        metrics["input_mode"] = exp["input_mode"]
        metrics["feature_type"] = exp["feature_type"]
        metrics["classifier_type"] = exp["classifier_type"]
        metrics["description"] = exp["description"]

        all_results.append(metrics)

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_pipeline = trained_pipeline
            best_experiment = exp

    # Save final summary CSV
    summary_path = output_dir / "final_results_summary.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "input_mode",
            "feature_type",
            "classifier_type",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "description",
        ])

        for row in all_results:
            writer.writerow([
                row["experiment"],
                row["input_mode"],
                row["feature_type"],
                row["classifier_type"],
                f'{row["accuracy"]:.4f}',
                f'{row["macro_f1"]:.4f}',
                f'{row["weighted_f1"]:.4f}',
                row["description"],
            ])

    # Save best final model
    dump(best_pipeline, output_dir / "best_final_pipeline.joblib")
    save_text(output_dir / "best_final_model.txt", best_experiment["name"])
    save_json(output_dir / "best_final_metrics.json", best_metrics)

    print(f"\n{'=' * 80}")
    print("Final experiments completed")
    print(f"{'=' * 80}")
    print("Results saved in:", output_dir)
    print("Summary:", summary_path)
    print("Best final model:", best_experiment["name"])
    print(f"Best macro F1: {best_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()