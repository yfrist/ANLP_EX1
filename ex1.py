"""Imports"""
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
import evaluate
import wandb


"""CLA Parser"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


"""Tokenize MRCP with Truncation (optionally limit dataset size - max_samples)"""
def preprocess_data(dataset, tokenizer, max_samples=-1):
    def tokenize(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    if max_samples != -1:
        dataset = dataset.select(range(max_samples))

    return dataset.map(tokenize, batched=True)


"""GLUE MRPC scorer (evaluates accuracy & F1)"""
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


"""Main"""
def main():
    args = parse_args()
    wandb.init(project="MRPC-finetune", config=vars(args))

    # Load dataset
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize
    tokenized_train = preprocess_data(raw_datasets["train"], tokenizer, args.max_train_samples)
    tokenized_eval = preprocess_data(raw_datasets["validation"], tokenizer, args.max_eval_samples)
    tokenized_test = preprocess_data(raw_datasets["test"], tokenizer, args.max_predict_samples)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path if args.model_path else "bert-base-uncased",
        num_labels=2
    )

    if args.do_train:
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            eval_strategy="epoch",  
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            report_to="wandb",
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )

        trainer.train()
        eval_result = trainer.evaluate()
        print("Validation accuracy:", eval_result["eval_accuracy"])

    if args.do_predict:
        model.eval()
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer)
        )

        predictions = trainer.predict(tokenized_test)
        preds = np.argmax(predictions.predictions, axis=-1)

        test_data = raw_datasets["test"]
        with open("predictions.txt", "w") as f:
            for i, pred in enumerate(preds):
                s1 = test_data[i]["sentence1"]
                s2 = test_data[i]["sentence2"]
                f.write(f"{s1}###{s2}###{pred}\n")


if __name__ == "__main__":
    main()
