# Finetuning
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Instruct
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

# Inference
from transformers import pipeline

model_checkpoint = "distilbert-base-uncased"

def instruct():
    instructModel = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(instructModel)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        instructModel,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    prompts = [
        "I will now provide you with a description of a module. Please generate 2-3 prompts that query which module is being described.\n\nModule description: Display and listing of version and copyright information.",
    ]
    
    for prompt in prompts:
        # https://huggingface.co/blog/llama2
        llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]
        """

        print(prompt)

        inputs = tokenizer(llamaPrompt, return_tensors="pt").to('cuda')

        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=512,
                top_p=1,
                temperature=0.01,
            )
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split("[/INST]", 1)[1].strip()

        print(response)

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
        evaluation_strategy = "steps",
        save_strategy = "steps",
        eval_steps=10000,
        save_steps=10000,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=30000,
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

def finetuneNoEval():
    dataset = Dataset.from_json('data/en_articles_classification_int.json')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=630)

    args = TrainingArguments(
        f"output/{model_checkpoint}-classification-noeval",
        save_strategy = "steps",
        save_steps=10000,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        max_steps=30000,
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
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def inference(modelName):
    model = AutoModelForSequenceClassification.from_pretrained(f"output/{modelName}")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    prompts = [
        "Which module provides version and copyright information?", # 0
        "How can I calculate the current time in another location?", # 627
        "Tell me how to test of the conversion of a temperature into the different heat units.", # 460
        "Where do I record both flexitime and operating data (BDE)?", # 626
        "Where can I check offer/order data?", # 608
        "Help me with inspection of partner data.", # 609
        "Provide me with resources on inspection of purchasing data.", # 610
        "Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.", # 45
    ]

    for prompt in prompts:
        print(classifier(prompt))

if __name__ == '__main__':
    instruct()

    #finetune()
    #finetuneNoEval()

    #inference("distilbert-base-uncased-classification/checkpoint-30000")
    #inference("distilbert-base-uncased-classification-noeval/checkpoint-30000")