"""Computes mean/median text and token length.

Care, this is a messy script which is based on preprocess.py.
"""

import re
import ujson

import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer
from tempfile import NamedTemporaryFile

from lxml.html import document_fromstring

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, StripAccents
from tokenizers.pre_tokenizers import Whitespace


RE_WHITESPACE = re.compile('\s+')


RE_UNWANTED_CHARACTERS_SVM = re.compile(
    # numbers, quotes, dashes, punctuation, brackets, slashes, other
    r'[0-9ʼ‘’´`„“”"\'\-––—\.,,?!:;…\(\)\[\]/\\~<>\|#\*&_+@]',
    flags=re.IGNORECASE
)


RE_UNWANTED_CHARACTERS = re.compile(
    # numbers, quotes, dashes, punctuation, brackets, slashes, other
    # but keep punctuation marks, hpyhen, and apostrophe
    r'[0-9ʼ‘’´`„“”"\––—\…\(\)\[\]/\\~<>\|#\*&_+@]',
    flags=re.IGNORECASE
)


# (U+2581) https://github.com/google/sentencepiece
SEPARATOR = '▁'

stemmer = PorterStemmer()
normalizer = normalizers.Sequence([NFKC(), StripAccents()])


def main(input_file):

    tmp_file = NamedTemporaryFile()
    pre_tokenizer = Whitespace()

    char_lengths = []
    token_lengths = []

    df = pd.read_csv(input_file, header=0, sep='\t')
    pos = 0
    neg = 0

    for _, row in df.iterrows():
            doc = row[df.columns.get_loc('content')]

            if row[df.columns.get_loc('target_label')] == 0:
                neg += 1
            elif row[df.columns.get_loc('target_label')] == 1:
                pos += 1

            char_lengths.append(len(doc))
            token_lengths.append(len(pre_tokenizer.pre_tokenize_str(doc)))

    print('instances:')
    print(f'  total: {len(df)}')
    print(f'  positive: {pos}')
    print(f'  negative: {neg}')

    print('text length:')
    print(f'  mean: {np.mean(char_lengths)}')
    print(f'  median: {np.median(char_lengths)}')

    print('token length:')
    print(f'  mean: {np.mean(token_lengths)}')
    print(f'  median: {np.median(token_lengths)}')

def document_fromstring_safe(text):
    if text is None:
        return ''
    else:
        return document_fromstring(text).text_content()


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help='path to the input data (tsv)')

    parser.add_argument('--svm', action='store_true',
                        help='applies svm-specific preprocessing (stemming) to the input data', default=False)

    args = parser.parse_args()

    main(args.input_file)
