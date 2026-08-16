import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

PROCESSED_PATH = settings.DATA_PROCESSED_DIR / "training_data.csv"
OUTPUT_DIR = settings.DATA_PROCESSED_DIR


def create_jsonl():
    if not PROCESSED_PATH.exists():
        print(f"File not found: {PROCESSED_PATH}")
        print("Please run clean_training_data.py first.")
        return

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {len(df)} rows from {PROCESSED_PATH}")

    # Create the prompt and completion pairs
    # We follow the same prompt format used by the Specialist Agent
    PREFIX = "Price is ₹"
    QUESTION = "What does this cost to the nearest rupee?"

    jsonl_data = []
    for _, row in df.iterrows():
        # Input format is the full_text that preprocessor generates
        text = str(row['full_text'])
        price = round(float(row['price']))

        prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
        completion = f"{price}.00"

        jsonl_data.append({
            "prompt": prompt,
            "completion": completion
        })

    # Split according to settings.FINETUNE_SPLIT (train, val, test fractions)
    train_frac, val_frac, test_frac = settings.FINETUNE_SPLIT
    train_data, temp_data = train_test_split(jsonl_data, test_size=(val_frac + test_frac), random_state=42)
    val_data, test_data = train_test_split(
        temp_data, test_size=test_frac / (val_frac + test_frac), random_state=42
    )

    def write_jsonl(data, filename):
        path = OUTPUT_DIR / filename
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Wrote {len(data)} rows to {path}")

    write_jsonl(train_data, 'finetune_train.jsonl')
    write_jsonl(val_data, 'finetune_val.jsonl')
    write_jsonl(test_data, 'finetune_test.jsonl')

if __name__ == "__main__":
    create_jsonl()
