import json
import sys
import os
import glob
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
)

MODEL_PATH = "deberta-conspiracy-classification"   # ✅ new folder
TEST_FILE = "dev_rehydrated.jsonl"
SUBMISSION_FILE = "submission.jsonl"
MODEL_NAME = "microsoft/deberta-v3-base"
LABEL_MAP = {0: "No", 1: "Yes"}
BATCH_SIZE = 32


def find_latest_checkpoint(base_path):
    ckpts = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not ckpts:
        print(f"Warning: no checkpoint-* folders in {base_path}, using base path.")
        return base_path
    ckpts.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
    latest = ckpts[-1]
    print(f"Found latest checkpoint: {latest}")
    return latest


def load_competition_test_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON at line {i}: {line}")
                continue
            sample_id = item.get("_id", f"sample_{i}")
            text = item.get("text", "")
            data.append({"unique_sample_id": sample_id, "text": text})
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_data(dataset, tokenizer):
    return dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True,
    )


if __name__ == "__main__":
    # 1) Load dev set for prediction
    raw_data = load_competition_test_data(TEST_FILE)
    if not raw_data:
        print("Error: no data loaded from dev file.")
        sys.exit(1)

    test_dataset = Dataset.from_list(raw_data)
    unique_ids = test_dataset["unique_sample_id"]

    # 2) Load tokenizer + latest checkpoint
    model_dir = find_latest_checkpoint(MODEL_PATH)
    print(f"Loading tokenizer from {MODEL_NAME} and model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 3) Tokenize
    tokenized_test = tokenize_data(test_dataset, tokenizer)
    tokenized_test = tokenized_test.remove_columns(["unique_sample_id", "text"])

    # 4) Trainer for inference
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_inference_deberta",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none",
        ),
        data_collator=data_collator,
    )

    # 5) Predict
    print("Starting prediction...")
    preds = trainer.predict(tokenized_test)
    logits = preds.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_labels = [LABEL_MAP[int(i)] for i in pred_ids]

    # 6) Write submission.jsonl
    print(f"Saving {len(pred_labels)} predictions to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, "w") as f:
        for uid, label in zip(unique_ids, pred_labels):
            obj = {"_id": uid, "conspiracy": label}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")
