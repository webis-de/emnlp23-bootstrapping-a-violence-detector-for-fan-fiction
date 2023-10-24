import csv
import pickle

import numpy as np
import pandas as pd

from pytorch_lightning import seed_everything
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from small_text.base import LABEL_UNLABELED
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
    TransformerModelArguments
)

from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer


BASE_MODEL = 'bert-base-uncased'
LR = 2e-5
CLASS_WEIGHT = None
NUM_EPOCHS = 10


def main(train_file, test_file, transformer_model, model_file, predictions_file):

    if 'longformer' not in transformer_model and 'bert' not in transformer_model:
        raise ValueError('unknown model')

    max_seq_len = 1024 if 'longformer' in model_file else 512
    batch_size = 4 if 'longformer' in model_file else 32

    df_train = pd.read_csv(train_file, sep='\t', quoting=csv.QUOTE_MINIMAL)
    df_test = pd.read_csv(test_file, sep='\t', quoting=csv.QUOTE_MINIMAL)

    ds_test, y_pred, clf, tokenizer = train_and_predict(df_train, df_test, max_seq_len,
                                                        transformer_model, batch_size=batch_size)

    pre_tokenizer = Whitespace()
    df_test['content_num_chars'] = df_test['content'].str.len()
    df_test['content_num_tokens'] = df_test['content'].apply(
        lambda content: len(pre_tokenizer.pre_tokenize_str(content)))
    df_test['prediction'] = y_pred
    df_test.loc[:, ].to_csv(predictions_file, index=False, sep='\t')

    y_true = df_test['target_label'].values

    with open(model_file, 'wb') as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred)

    print(f'ACC: {acc}')
    print(f'BACC: {bacc}')
    print(f'PREC: {prec}')
    print(f'REC: {rec}')
    print(f'F1: {fscore}')
    print('')
    print(classification_report(y_true, y_pred, labels=[0, 1]))


def train_and_predict(df_train, df_test, max_seq_len, model, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    ds_train = build_transformers_dataset(df_train, tokenizer, max_seq_length=max_seq_len)
    ds_test = build_transformers_dataset(df_test, tokenizer, max_seq_length=max_seq_len)

    clf = TransformerBasedClassification(TransformerModelArguments(model,
                                                                   tokenizer=BASE_MODEL,
                                                                   config=model),
                                         num_epochs=NUM_EPOCHS,
                                         num_classes=2,
                                         model_selection=False,
                                         class_weight=CLASS_WEIGHT,
                                         validations_per_epoch=5,
                                         mini_batch_size=batch_size,
                                         validation_set_size=0.1,
                                         lr=LR)
    clf.fit(ds_train)

    y_pred = clf.predict(ds_test)

    return ds_test, y_pred, clf, tokenizer


def build_transformers_dataset(df_data, tokenizer, max_seq_length=512,
                               target_labels=np.array([0, 1]), unlabeled=False):

    text = df_data['content'].values

    if unlabeled:
        labels = [LABEL_UNLABELED] * len(df_data)
    else:
        labels = df_data['target_label'].values

    data_out = []

    for i in range(len(df_data)):
        encoded_dict = tokenizer.encode_plus(
            text[i],
            add_special_tokens=True,
            padding='max_length',
            max_length=max_seq_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))
    if unlabeled:
        return TransformersDataset(data_out, target_labels=target_labels)
    else:
        return TransformersDataset(data_out)


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    seed_everything(42)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('train_file', help='input file with one document per line (tsv)')
    parser.add_argument('test_file', help='input file with one document per line (tsv)')
    # model must be bert-base-uncased or continued from bert-base-uncased
    parser.add_argument('transformer_model', help='transformer model to use (name or folder)')
    parser.add_argument('model_file', help='output file where the model is saved to (pkl)')
    parser.add_argument('predictions_file', help='output file predictions file is saved (tsv)')

    args = parser.parse_args()

    main(args.train_file, args.test_file, args.transformer_model, args.model_file,
         args.predictions_file)
