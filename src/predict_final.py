import json
import csv
from pathlib import Path

from joblib import load


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_input_text(sample):
    """
    Best final model input:
    current_text only.

    This must match the training input used in tune_best_svm.py.
    """
    return sample.get("current_text", "").strip()


def main():
    project_root = Path(__file__).resolve().parent.parent

    test_path = project_root / "input_data" / "test.json"
    outputs_dir = project_root / "outputs_final"

    best_model_path = outputs_dir / "best_final_pipeline.joblib"
    best_model_name_path = outputs_dir / "best_final_model.txt"
    best_metrics_path = outputs_dir / "best_final_metrics.json"

    if not test_path.exists():
        print("Error: test.json not found at:", test_path)
        return

    if not best_model_path.exists():
        print("Error: best_final_pipeline.joblib not found at:", best_model_path)
        print("Run src/tune_best_svm.py first.")
        return

    test_data = load_json(test_path)
    print(f"Loaded test samples: {len(test_data)}")

    model = load(best_model_path)

    if best_model_name_path.exists():
        with open(best_model_name_path, "r", encoding="utf-8") as f:
            best_model_name = f.read().strip()
        print("Best final model:", best_model_name)

    if best_metrics_path.exists():
        best_metrics = load_json(best_metrics_path)
        print("Best validation metrics:")
        print(f"  Accuracy    : {best_metrics.get('accuracy'):.4f}")
        print(f"  Macro F1    : {best_metrics.get('macro_f1'):.4f}")
        print(f"  Weighted F1 : {best_metrics.get('weighted_f1'):.4f}")

    test_texts = [build_input_text(sample) for sample in test_data]

    predictions = model.predict(test_texts)

    prediction_data = []
    for sample, pred in zip(test_data, predictions):
        new_sample = dict(sample)
        new_sample["label"] = int(pred)
        prediction_data.append(new_sample)

    prediction_json_path = outputs_dir / "prediction.json"
    save_json(prediction_json_path, prediction_data)

    prediction_csv_path = outputs_dir / "test_predictions.csv"
    with open(prediction_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dialogue_id", "current_text", "predicted_label"])

        for sample, pred in zip(test_data, predictions):
            writer.writerow([
                sample.get("id"),
                sample.get("dialogue_id"),
                sample.get("current_text"),
                int(pred),
            ])

    print("\nFinal prediction completed.")
    print("Saved:", prediction_json_path)
    print("Saved:", prediction_csv_path)
    print("Submit prediction.json if the competition expects JSON format.")


if __name__ == "__main__":
    main()