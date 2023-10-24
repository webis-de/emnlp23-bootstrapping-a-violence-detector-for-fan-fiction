import csv
import pickle

import numpy as np
import pandas as pd


def main(test_file, output_file, model_file):
    df = pd.read_csv(test_file, sep='\t', quoting=csv.QUOTE_MINIMAL)

    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    x_test = clf.vectorizer.transform(df['content'].values)

    proba = clf.predict_proba(x_test)
    df['prediction'] = np.argmax(proba, axis=1)
    df['confidence'] = proba.max(axis=1)

    del df['content'], df['title']

    df.to_csv(output_file, index=False, sep='\t')


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('test_file', help='input file with one document per line (tsv)')
    parser.add_argument('output_file', help='output file without content '
                                            'predictions and confidence estimates (tsv)')
    parser.add_argument('model_file', help='output file where the model is saved to (pkl)')

    args = parser.parse_args()

    main(args.test_file, args.output_file, args.model_file)
