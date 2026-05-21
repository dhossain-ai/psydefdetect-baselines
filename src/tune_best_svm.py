import json
import csv
from pathlib import Path

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


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


def build_input_text(sample):
    # Best previous input mode: current_text only
    return sample.get("current_text", "").strip()


def prepare_data(data):
    texts = []
    labels = []

    for sample in data:
        texts.append(build_input_text(sample))
        labels.append(sample["label"])

    return texts, labels


def make_pipeline(c_value):
    return Pipeline([
        ("features", TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=50000,
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            C=c_value,
            max_iter=10000,
        )),
    ])


def main():
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "input_data" / "train.json"

    output_dir = project_root / "outputs_final" / "svm_c_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_json(train_path)
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

    x_train, y_train = prepare_data(train_samples)
    x_val, y_val = prepare_data(val_samples)

    c_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    all_results = []
    best_metrics = None
    best_pipeline = None
    best_c = None

    for c_value in c_values:
        print(f"\nTraining current_text + TF-IDF 1-3 + LinearSVC, C={c_value}")

        pipeline = make_pipeline(c_value)
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_val)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        weighted_f1 = f1_score(y_val, y_pred, average="weighted")

        print(f"Accuracy    : {acc:.4f}")
        print(f"Macro F1    : {macro_f1:.4f}")
        print(f"Weighted F1 : {weighted_f1:.4f}")

        metrics = {
            "experiment": f"current_text_word_1_3_svm_C_{c_value}",
            "input_mode": "current_text",
            "feature_type": "word_1_3",
            "classifier": "LinearSVC",
            "C": c_value,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

        all_results.append(metrics)

        report_text = classification_report(
            y_val,
            y_pred,
            digits=4,
            zero_division=0,
        )

        safe_c = str(c_value).replace(".", "_")
        save_text(output_dir / f"classification_report_C_{safe_c}.txt", report_text)

        dump(pipeline, output_dir / f"pipeline_C_{safe_c}.joblib")

        if best_metrics is None or macro_f1 > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_pipeline = pipeline
            best_c = c_value

    summary_path = output_dir / "svm_c_tuning_summary.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "input_mode",
            "feature_type",
            "classifier",
            "C",
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ])

        for row in all_results:
            writer.writerow([
                row["experiment"],
                row["input_mode"],
                row["feature_type"],
                row["classifier"],
                row["C"],
                f'{row["accuracy"]:.4f}',
                f'{row["macro_f1"]:.4f}',
                f'{row["weighted_f1"]:.4f}',
            ])

    save_json(output_dir / "best_svm_c_metrics.json", best_metrics)
    save_text(output_dir / "best_svm_c.txt", str(best_c))
    dump(best_pipeline, output_dir / "best_svm_c_pipeline.joblib")

    # Also copy this as the global best final model if it beats previous best.
    global_best_metrics_path = project_root / "outputs_final" / "best_final_metrics.json"

    should_replace_global = True
    if global_best_metrics_path.exists():
        previous_best = load_json(global_best_metrics_path)
        previous_macro_f1 = previous_best.get("macro_f1", -1)
        should_replace_global = best_metrics["macro_f1"] >= previous_macro_f1

    if should_replace_global:
        dump(best_pipeline, project_root / "outputs_final" / "best_final_pipeline.joblib")
        save_text(project_root / "outputs_final" / "best_final_model.txt", best_metrics["experiment"])
        save_json(project_root / "outputs_final" / "best_final_metrics.json", best_metrics)

        print("\nUpdated global best final model.")
    else:
        print("\nDid not update global best final model.")

    print("\nTuning completed.")
    print("Best C:", best_c)
    print(f"Best Macro F1: {best_metrics['macro_f1']:.4f}")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()