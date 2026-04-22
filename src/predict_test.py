import json
import csv
from pathlib import Path

from joblib import load


# This function reads a json file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# This function saves json
def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# This function builds the input text exactly like training
# Very important: test input must be prepared in the same way
def build_input_text(sample):
    turns = sample.get("dialogue", [])
    parts = []

    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        parts.append(f"{speaker}: {text}")

    return " ".join(parts)


def main():
    # Project root = one folder above src
    project_root = Path(__file__).resolve().parent.parent

    test_path = project_root / "input_data" / "test.json"
    outputs_dir = project_root / "outputs"

    best_model_path = outputs_dir / "best_pipeline.joblib"
    best_model_name_path = outputs_dir / "best_model.txt"

    # Check files exist
    if not test_path.exists():
        print("Error: test.json not found.")
        return

    if not best_model_path.exists():
        print("Error: best_pipeline.joblib not found.")
        print("Run run_baselines.py first.")
        return

    # Load test data
    test_data = load_json(test_path)
    print(f"Loaded {len(test_data)} test samples.")

    # Load best model
    model = load(best_model_path)

    if best_model_name_path.exists():
        with open(best_model_name_path, "r", encoding="utf-8") as f:
            best_model_name = f.read().strip()
        print("Best model:", best_model_name)

    # Prepare test texts
    test_texts = [build_input_text(sample) for sample in test_data]

    # Predict labels
    predictions = model.predict(test_texts)

    # Create final prediction json
    # We copy each test sample and add the predicted label
    prediction_data = []
    for sample, pred in zip(test_data, predictions):
        new_sample = dict(sample)
        new_sample["label"] = int(pred)
        prediction_data.append(new_sample)

    # Save official-style prediction file
    prediction_json_path = outputs_dir / "prediction.json"
    save_json(prediction_json_path, prediction_data)

    # Save a small csv also for easy checking
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

    print("\nDone")
    print("Saved:", prediction_json_path)
    print("Saved:", prediction_csv_path)
    print("Now check prediction.json before final submission.")


if __name__ == "__main__":
    main()