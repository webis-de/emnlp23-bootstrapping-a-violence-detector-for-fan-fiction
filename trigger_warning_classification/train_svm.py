import csv
import pickle

import numpy as np
import pandas as pd

from pytorch_lightning import seed_everything
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from small_text.classifiers import ConfidenceEnhancedLinearSVC

from tokenizers.pre_tokenizers import Whitespace
from trigger_warning_classification.stopwords.stopwords import get_stopwords


# best settings
C = 0.5
NORM = 'l2'
BINARY = True
NGRAMS = 2
LOWERCASE = True
CLASS_WEIGHT = 'balanced'
MAX_FEATURES = 100_000


def main(train_file, test_file, model_file, predictions_file):

    df_train = pd.read_csv(train_file, sep='\t', quoting=csv.QUOTE_MINIMAL)
    df_test = pd.read_csv(test_file, sep='\t', quoting=csv.QUOTE_MINIMAL)

    clf, vectorizer, y_pred = train_and_predict(df_train, df_test)

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

    n = 3000
    inverse_dict = dict({
        index:term for term, index in vectorizer.vocabulary_.items()
    })
    print(f'\n\n{n} best positive features:\n')
    for i in np.argsort(-clf.coef_[0, :])[:n]:
        print(f'{inverse_dict[i]}\t{clf.coef_[0, i]}')

    print(f'\n\n{n} best negative features:\n')
    for i in np.argsort(clf.coef_[0, :])[:n]:
        print(f'{inverse_dict[i]}\t{clf.coef_[0, i]}')


def train_and_predict(df_train, df_test):
    vectorizer = TfidfVectorizer(ngram_range=(1, NGRAMS),
                                 lowercase=LOWERCASE,
                                 max_features=MAX_FEATURES,
                                 min_df=2,
                                 norm=NORM,
                                 binary=BINARY,
                                 stop_words=get_stopwords(),
                                 token_pattern=r'(?u)\b\w\w\w+\b')

    x_train = vectorizer.fit_transform(df_train['content'].values)
    x_test = vectorizer.transform(df_test['content'].values)

    clf = ConfidenceEnhancedLinearSVC(linearsvc_kwargs={'C': C, 'class_weight': CLASS_WEIGHT})
    clf.fit(x_train, df_train['target_label'].values)
    y_pred = clf.predict(x_test)

    # attach vectorizer to classifier for easier serialization
    clf.vectorizer = vectorizer

    return clf, vectorizer, y_pred


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
    parser.add_argument('model_file', help='output file where the model is saved to (pkl)')
    parser.add_argument('predictions_file', help='output file predictions file is saved (tsv)')

    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.predictions_file)
