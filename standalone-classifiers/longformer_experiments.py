import json
import logging

import datasets
from tqdm import tqdm
import click

from pathlib import Path

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import transformers
import code
import torch
from models import get_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


import numpy as np

transformers.logging.set_verbosity_info()
metric = evaluate.load("accuracy")


def _load_dataset(dataset_path, data_name, checkpoint, validate=False, training=True):
    """
    return: tokenizer, train, validation, test
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    f_label = datasets.ClassLabel(num_classes=2, names=["non-violent", 'violent'])

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True)

    dataset_test = load_dataset("json", data_files=str((dataset_path / f"{data_name}-test.jsonl").resolve()))
    dataset_test = dataset_test.cast_column('label', f_label)
    dataset_test.set_format("torch")
    tokenized_data_test = (dataset_test.map(tokenize_function, batched=True, remove_columns=['id', 'text']))

    if training:
        dataset_train = load_dataset("json", data_files={"train": str((dataset_path / f"{data_name}-train.jsonl").resolve())})
        f_label = datasets.ClassLabel(num_classes=2, names=["non-violent", 'violent'])
        dataset_train = dataset_train.cast_column('label', f_label)
        dataset_train.set_format("torch")

        if validate:
            dataset_train = dataset_train["train"].train_test_split(test_size=0.1, stratify_by_column="label")

        tokenized_data_train = (dataset_train.map(tokenize_function, batched=True, remove_columns=['id', 'text']))

        return tokenizer, tokenized_data_train["train"], tokenized_data_test['train'], \
               tokenized_data_train["test"] if validate else None

    return tokenizer, None, tokenized_data_test['train'], None


def _evaluate(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def run_experiment(dataset_path: str, sample_name: str, checkpoint: str, savepoint: str, epochs=20, batches=4,
                   lr=0.0005, model_type='longformer', validate=True):
    logging.warning("load dataset and tokenizer")
    dataset_path = Path(dataset_path)
    tok, ds_train, ds_test, ds_validation = _load_dataset(dataset_path, sample_name, checkpoint)
    logging.warning("start model training")

    num_labels = 2

    # TODO https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
    # Params following https://github.com/amazon-science/efficient-longdoc-classification
    args = TrainingArguments(
        output_dir=savepoint,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batches,
        per_device_eval_batch_size=batches,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=False,
        report_to="wandb",
        run_name=f"violence-{model_type}-{sample_name}")

    model = get_model(checkpoint, num_labels, model_type)
    trainer = Trainer(model=model, args=args,
                      train_dataset=ds_train, eval_dataset=ds_test,
                      tokenizer=tok, compute_metrics=_evaluate)

    trainer.evaluate()

    # TODO https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=NboJ7kDOIrJq
    # best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
    # logging.info(f"best parameters found: {best_run.hyperparameters.items()}")
    # for n, v in best_run.hyperparameters.items():
    #     setattr(trainer.args, n, v)
    trainer.train()

    logging.warning("save trained model")
    trainer.save_model(savepoint)

    predictions = trainer.predict(ds_test)

    results = getattr(predictions, "metrics")
    open(f"{savepoint}/results.json", 'w').write(json.dumps(results))
    return savepoint


def make_and_save_predictions(dataset_path: str, name: str, model_checkpoint: str, tokenizer_checkpoint: str,
                              savepoint: str):
    logging.warning("load dataset")
    f_label = datasets.ClassLabel(num_classes=2, names=["non-violent", 'violent'])
    tokenizer_kwargs = {'padding': True, 'truncation': True}

    dataset_test = load_dataset("json", data_files=f"{dataset_path}/{name}-test.jsonl")
    dataset_test = dataset_test.cast_column('label', f_label)
    dataset_test.set_format("torch")

    logging.warning("load pipeline")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    classifier = pipeline(model=model_checkpoint, tokenizer=tokenizer, task="text-classification",
                          device="cuda:0")

    def add_length(example):
        return {"length": len(example["text"].split(" "))}

    def classify(examples):
        return {"prediction": classifier(examples["text"], **tokenizer_kwargs)}

    logging.warning("make predictions")
    dataset_with_length = dataset_test.map(add_length)
    dataset_with_predictions = dataset_with_length.map(classify, batched=True, batch_size=32, remove_columns=["text"])
    for split, data in dataset_with_predictions.items():
        data.to_json(f"{savepoint}/predictions-{split}.jsonl")

    ## Evaluation of results
    ds_test = dataset_with_predictions["train"]
    truth = ds_test["label"]
    predictions = [1 if p["label"] == "LABEL_1" else 0 for p in ds_test["prediction"]]

    def _scores(t, p):
        return {"f1": round(f1_score(t, p), 3), "p": round(precision_score(t, p), 3),
                "r": round(recall_score(t, p), 3), "acc": round(accuracy_score(t, p), 3)}

    results = _scores(truth, predictions)

    for lower, upper in {(0, 512), (512, 4096), (4096, 16000), (16000, 10e+5)}:
        results[f"{lower}-{upper}"] = _scores([t for t, l in zip(truth, ds_test["length"]) if lower < l <= upper],
                                              [p for p, l in zip(predictions, ds_test["length"]) if lower < l <= upper])

    open(f"{savepoint}/prediction-metrics.json", 'w').write(json.dumps(results))


@click.group()
def cli():
    pass


@click.option('-d', '--dataset-dir', type=click.Path(exists=True), help="Path where the xxx-train.jsonl and xxx-test.jsonl is.")
@click.option('-n', '--name', type=click.STRING, default='develop', help="base name of the sample (i.e. `random-balanced`)")
@click.option('-m', '--model_type', type=click.STRING, default='longformer', help="which model to use (see models.py)")
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the model in.")
@click.option('--epochs', type=click.INT, default=20, help="Number of training epochs")
@click.option('--batches', type=click.INT, default=4, help="Batch size")
@click.option('--lr', type=click.FLOAT, default=0.0005, help="Initial learning rate")
@cli.command()
def run(dataset_dir, name, model_type, savepoint, epochs, batches, lr):
    """
    This is the CLI interface to train the model .

      python3 /mnt/ceph/storage/data-tmp/2022/mike4537/trigger-warning-classification/standalone-classifiers/longformer_experiments.py run \
        -n "random-balanced" \
        -d "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/jsonl_datasets/v3-arr23/normal" \
        -s "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/violence-classification/classification-results/arr23-submission/longformer/random-balanced"
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)

    if model_type == 'longformer':
        checkpoint = 'allenai/longformer-base-4096'
    else:
        checkpoint = "bert-base-uncased"

    run_experiment(dataset_dir, name, checkpoint, savepoint, epochs=epochs, batches=batches, lr=lr, model_type=model_type)
    print(f"trained model at {savepoint}")


@click.option('-c', '--checkpoint', type=click.STRING, default='allenai/longformer-base-4096', help="base checkpoint for model and tokenized")
@click.option('-d', '--dataset-dir', type=click.Path(exists=True), help="Path where the training.jsonl and validation.jsonl is.")
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the model in.")
@click.option('--epochs', type=click.INT, default=3, help="Path to save the model in.")
@cli.command()
def hyperparameters(checkpoint, dataset_dir, savepoint, epochs):
    """ TODO hyperparameter sweep (cf. the comments in the train loop). May also be a flag in the run method.  """
    pass


@click.option('-c', '--checkpoint', type=click.STRING, help="checkpoint to load")
@click.option('-n', '--name', type=click.STRING, default='develop', help="base name of the sample (i.e. `random-balanced`)")
@click.option('-d', '--dataset-dir', type=click.Path(exists=True), help="Path where the training.jsonl and validation.jsonl is.")
@click.option('-s', '--savepoint', type=click.Path(), help="Path to save the predictions in.")
@click.option('-m', '--model_type', type=click.STRING, default='longformer', help="which model to use (see models.py)")
@cli.command()
def predict(checkpoint, name, dataset_dir, savepoint, model_type):
    """
    How to run `predict` from cli (example):

         python3 /mnt/ceph/storage/data-tmp/2022/mike4537/trigger-warning-classification/standalone-classifiers/longformer_experiments.py predict \
            -n "tag-frequency-balanced" \
            -c "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/violence-classification/classification-results/arr23-submission/longformer/tag-frequency-balanced" \
            -d "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/jsonl_datasets/v3-arr23/normal" \
            -s "/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/violence-classification/classification-results/arr23-submission/longformer/tag-frequency-balanced"
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)

    if model_type == 'longformer':
        tokenizer_checkpoint = 'allenai/longformer-base-4096'
    else:
        tokenizer_checkpoint = "bert-base-uncased"
    make_and_save_predictions(dataset_dir, name, checkpoint, tokenizer_checkpoint, savepoint)


if __name__ == "__main__":
    cli()
