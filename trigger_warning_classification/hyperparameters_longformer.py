import csv
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
    TransformerModelArguments
)
from trigger_warning_classification.train_bert import build_transformers_dataset


BASE_MODEL = 'allenai/longformer-base-4096'


def main(train_file, output_file, validation_set_size=0.1):
    df = pd.read_csv(train_file, sep='\t', quoting=csv.QUOTE_MINIMAL)

    split = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_size)
    indices_train, indices_valid = \
    list(split.split(np.zeros(len(df)), df['target_label'].astype(int).values))[0]

    df_train = df.iloc[indices_train]
    df_valid = df.iloc[indices_valid]

    y_true = df_valid['target_label'].values

    with open(output_file, 'w+') as f:
        f.write('ACC\tB_ACC\tPREC\tREC\tF1\t'
                'num_epochs\tlr\tclass_weight\n')
        evaluate_hyperparameters(df_train, df_valid, f, y_true)


def evaluate_hyperparameters(df_train, df_valid, f, y_true):

    max_seq_len = 1024

    for num_epochs in [5, 10, 15]:
        for lr in [1e-5, 2e-5, 5e-5, 1e-4]:
            for class_weight in [None, 'balanced']:

                ds_test, y_pred, _, _ = train_predict_bert(
                    df_train,
                    df_valid,
                    max_seq_len,
                    BASE_MODEL,
                    num_epochs,
                    lr,
                    class_weight
                )

                acc = accuracy_score(y_true, y_pred)
                bacc = balanced_accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                fscore = f1_score(y_true, y_pred)

                print('+++\n')
                f.write(f'{acc}\t{bacc}\t{prec}\t{rec}\t{fscore}\t'
                        f'{num_epochs}\t{lr}\t'
                        f'{class_weight}\n')
                f.flush()


def train_predict_bert(df_train, df_test, max_seq_len, model, num_epochs, lr, class_weight, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    ds_train = build_transformers_dataset(df_train, tokenizer, max_seq_length=max_seq_len)
    ds_test = build_transformers_dataset(df_test, tokenizer, max_seq_length=max_seq_len)

    clf = TransformerBasedClassification(TransformerModelArguments(model,
                                                                   tokenizer=BASE_MODEL,
                                                                   config=model),
                                         num_epochs=num_epochs,
                                         num_classes=2,
                                         model_selection=False,
                                         class_weight=class_weight,
                                         validations_per_epoch=5,
                                         mini_batch_size=batch_size,
                                         validation_set_size=0.1,
                                         lr=lr)
    clf.fit(ds_train)


    y_pred = clf.predict(ds_test)

    return ds_test, y_pred, clf, tokenizer


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('train_file', help='input file with one document per line (tsv)')
    parser.add_argument('output_file', help='output file with the measured results (tsv)')

    args = parser.parse_args()

    main(args.train_file, args.output_file)
