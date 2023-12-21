import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model_checkpoint = "distilbert-base-uncased"

def finetune():
    full_dataset = Dataset.from_json('data/en_articles_classification_int.json')
    dataset = full_dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=630)

    args = TrainingArguments(
        f"output/{model_checkpoint}-classification",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "prf": precision_recall_fscore_support(labels, predictions, average="macro"),
        }

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

from transformers import pipeline

def inference():
    model = AutoModelForSequenceClassification.from_pretrained("output/distilbert-base-uncased-classification/checkpoint-2835")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    result = classifier("Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.")
    print(result)

if __name__ == '__main__':
    #finetune()
    inference()