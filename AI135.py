from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

# Load the dataset (e.g., a CSV file)
train_dataset = load_dataset("csv", data_files="train.csv")
test_dataset = load_dataset("csv", data_files="test.csv")

# Tokenize the text data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_train = tokenizer(train_dataset["text"], truncation=True, padding=True)
tokenized_test = tokenizer(test_dataset["text"], truncation=True, padding=True)

# Convert the tokenized data to PyTorch datasets
train_dataset = Dataset.from_dict(tokenized_train)
test_dataset = Dataset.from_dict(tokenized_test)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
