import csv
import re
import ujson

import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer
from tempfile import NamedTemporaryFile

from lxml.html import document_fromstring
from pytorch_lightning import seed_everything
from resiliparse.parse.lang import detect_fast
from sklearn.model_selection import StratifiedShuffleSplit

from textacy.preprocessing.replace import urls as replace_urls, emojis as replace_emojis
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, StripAccents
from tokenizers.pre_tokenizers import Whitespace


MIN_NUM_TOKENS = 10


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


def main(input_file, output_train_file, output_test_file, test_set_size=0.1, svm=False):

    tmp_file = NamedTemporaryFile()
    dropped = 0

    with open(input_file, 'r') as f_in, open(tmp_file.name, 'w+') as f_out:
        writer = csv.writer(f_out, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['work_id', 'tags', 'target_label', 'title', 'content'])
        pre_tokenizer = Whitespace()
        for line in f_in:
            doc = ujson.loads(line.strip())
            texts = [
                preprocess_text(document_fromstring_safe(chapter['content']),
                                svm=svm)
                for chapter in doc['chapters']
            ]
            tokens = [
                token
                for text in texts
                for token, _ in pre_tokenizer.pre_tokenize_str(text)
            ]
            content = ' '.join(texts)

            if len(tokens) >= MIN_NUM_TOKENS:
                language, out_of_place = detect_fast(content)
                if language == 'en' and out_of_place < 1200:
                    doc = [
                        str(doc['work_id']).replace('\t', ' '),
                        '▁'.join([tag['name'] for tag in doc['tags']]).replace('\t', ' '),
                        str(int('Graphic Depictions Of Violence' in set(tag['name'] for tag in doc['tags']))),
                        doc['title'].replace('\t', ' '),
                        content
                    ]
                    writer.writerow(doc)
            else:
                dropped += 1
                print(content)

        print(f'Dropped {dropped} samples (non-English text)')

        df = pd.read_csv(tmp_file.name, sep='\t', quoting=csv.QUOTE_MINIMAL)

        if output_test_file is not None:
            df.reindex(np.random.permutation(df.index))

            split = StratifiedShuffleSplit(n_splits=1, test_size=test_set_size)
            indices_train, indices_test = list(split.split(np.zeros(len(df)), df['target_label'].astype(int).values))[0]

            df_train = df.iloc[indices_train]
            df_train.to_csv(output_train_file, index=False, sep='\t')

            df_test = df.iloc[indices_test]
            df_test.to_csv(output_test_file, index=False, sep='\t')
        else:
            df.to_csv(output_train_file, index=False, sep='\t')


def document_fromstring_safe(text):
    if text is None:
        return ''
    else:
        return document_fromstring(text).text_content()


def preprocess_text(text, svm=False):
    text = normalizer.normalize_str(text)
    text = replace_emojis(text, ' ')
    if svm is True:
        text = replace_urls(text, 'CONTAINS' + SEPARATOR + 'URL')
        text = RE_UNWANTED_CHARACTERS_SVM.sub(' ', text)
    else:
        text = replace_urls(text, 'URL')
        text = RE_UNWANTED_CHARACTERS.sub(' ', text)
    if svm is True:
        text = ' '.join(map(lambda x: stemmer.stem(x), text.split(' ')))
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    return text


if __name__ == '__main__':
    import argparse
    import logging
    import sys

    seed_everything(42)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help='path to the input data (jsonlines)')
    parser.add_argument('output_train_file', help='path to the output train file (tsv)')
    parser.add_argument('output_test_file', nargs='?', help='[optional!] path to the output test file (tsv)')

    parser.add_argument('--svm', action='store_true',
                        help='applies svm-specific preprocessing (stemming) to the input data', default=False)

    args = parser.parse_args()

    main(args.input_file, args.output_train_file, args.output_test_file, svm=args.svm)
