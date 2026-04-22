import json
from pathlib import Path
from collections import Counter


# This function reads a json file and returns the data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# This function prints one sample in a clean way
def print_sample(sample, title="Sample"):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    print("id:", sample.get("id"))
    print("dialogue_id:", sample.get("dialogue_id"))
    print("current_text:", sample.get("current_text"))

    if "label" in sample:
        print("label:", sample.get("label"))

    print("\ndialogue:")
    for i, turn in enumerate(sample.get("dialogue", []), start=1):
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        print(f"  {i}. [{speaker}] {text}")


# This function prints basic info about one dataset
def inspect_dataset(data, name="dataset"):
    print(f"\n{'=' * 60}")
    print(f"Inspecting {name}")
    print(f"{'=' * 60}")

    # Total number of examples
    print("Number of samples:", len(data))

    if len(data) == 0:
        print("Dataset is empty.")
        return

    # Show all keys in the first sample
    first_sample = data[0]
    print("Fields in one sample:", list(first_sample.keys()))

    # Count unique dialogue ids
    dialogue_ids = [item.get("dialogue_id") for item in data if "dialogue_id" in item]
    print("Number of unique dialogue ids:", len(set(dialogue_ids)))

    # Count dialogue lengths
    dialogue_lengths = [len(item.get("dialogue", [])) for item in data]
    print("Min dialogue length:", min(dialogue_lengths))
    print("Max dialogue length:", max(dialogue_lengths))
    print("Average dialogue length:", round(sum(dialogue_lengths) / len(dialogue_lengths), 2))

    # If labels exist, print label counts
    if "label" in first_sample:
        labels = [item["label"] for item in data]
        label_counts = Counter(labels)

        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  label {label}: {count}")

    # Print first sample
    print_sample(first_sample, title=f"First sample from {name}")


def main():
    # Project root = one folder above src
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "input_data"

    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    # Check files exist
    if not train_path.exists():
        print("Error: train.json not found at:", train_path)
        return

    if not test_path.exists():
        print("Error: test.json not found at:", test_path)
        return

    # Load files
    train_data = load_json(train_path)
    test_data = load_json(test_path)

    # Inspect both
    inspect_dataset(train_data, name="train.json")
    inspect_dataset(test_data, name="test.json")

    # Print one extra example from train if possible
    if len(train_data) > 1:
        print_sample(train_data[1], title="Second sample from train.json")

    # Print one extra example from test if possible
    if len(test_data) > 1:
        print_sample(test_data[1], title="Second sample from test.json")


if __name__ == "__main__":
    main()