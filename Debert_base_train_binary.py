import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def load_and_filter_data(file_path):
    """Loads data from a JSONL file and keeps only Yes/No conspiracy labels."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue
            if item.get("conspiracy") in ["Yes", "No"] and "text" in item:
                data.append(item)
    print(f"Loaded {len(data)} training examples with Yes/No labels.")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data."""
    return dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True
    )


def encode_labels(dataset, label_to_id):
    """Encodes string labels to integer ids."""
    return dataset.map(
        lambda examples: {"labels": [label_to_id[l] for l in examples["conspiracy"]]},
        batched=True
    )


if __name__ == "__main__":
    train_file = "train_rehydrated.jsonl"

    # ✅ Use DeBERTa (base size to avoid OOM)
    model_name = "microsoft/deberta-v3-base"
    output_dir = "deberta-conspiracy-classification"

    label_to_id = {"No": 0, "Yes": 1}
    id_to_label = {0: "No", 1: "Yes"}
    num_labels = len(label_to_id)

    # 🔻 Smaller batch size to reduce MPS memory usage
    batch_size = 4
    learning_rate = 2e-5
    num_epochs = 5

    # 1) Load and filter data
    train_data = load_and_filter_data(train_file)
    train_dataset = Dataset.from_list(train_data)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Tokenize + encode labels
    tokenized_train = tokenize_data(train_dataset, tokenizer)
    encoded_train = encode_labels(tokenized_train, label_to_id)

    # 4) Load DeBERTa model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # 5) Training arguments (simple, no eval to save memory)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs_deberta",
        report_to="none",
        save_strategy="epoch",     # save checkpoints each epoch
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        tokenizer=tokenizer,
    )

    print("Training DeBERTa model...")
    trainer.train()
    print("Training finished.")
