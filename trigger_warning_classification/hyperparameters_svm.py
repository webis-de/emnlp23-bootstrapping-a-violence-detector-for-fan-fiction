import csv
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC


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
                'c\tngrams\tnorm\tbinary\tlc\tclass_weights\tmax_features\n')
        evaluate_hyperparameters(df_train, df_valid, f, y_true)


def evaluate_hyperparameters(df_train, df_valid, f, y_true):

    for c in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for norm in ['l2']:
            for binary in [True, False]:
                for ngrams in range(1, 3):
                    for lowercase in [True, False]:
                        for class_weight in ['balanced', None]:
                            for max_features in [25000, 50000, 100000, None]:
                                x_test, y_pred, clf, vectorizer = train_predict_sklearn(
                                    df_train,
                                    df_valid,
                                    c,
                                    norm,
                                    binary,
                                    (1, ngrams),
                                    lowercase,
                                    class_weight,
                                    max_features
                                )

                                acc = accuracy_score(y_true, y_pred)
                                bacc = balanced_accuracy_score(y_true, y_pred)
                                prec = precision_score(y_true, y_pred)
                                rec = recall_score(y_true, y_pred)
                                fscore = f1_score(y_true, y_pred)

                                print('+++\n')
                                f.write(f'{acc}\t{bacc}\t{prec}\t{rec}\t{fscore}\t'
                                        f'{c}\t{ngrams}\t{norm}\t{binary}\t{lowercase}\t'
                                        f'{class_weight}\t{max_features}\n')
                                f.flush()


def train_predict_sklearn(df_train, df_test, c, norm, binary, ngram_range, lowercase,
                          class_weight, max_features):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                 lowercase=lowercase,
                                 max_features=max_features,
                                 min_df=2,
                                 norm=norm,
                                 binary=binary)

    x_train = vectorizer.fit_transform(df_train['content'].values)
    x_test = vectorizer.transform(df_test['content'].values)

    clf = LinearSVC(class_weight=class_weight, C=c)

    clf.fit(x_train, df_train['target_label'].values)
    y_pred = clf.predict(x_test)

    return x_test, y_pred, clf, vectorizer


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
