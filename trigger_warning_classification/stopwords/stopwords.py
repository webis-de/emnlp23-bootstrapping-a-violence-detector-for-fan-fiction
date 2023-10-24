import stopwordsiso as stopwordsiso

from pathlib import Path
from spacy.lang.en.stop_words import STOP_WORDS


def get_stopwords():
    stopwords = []

    with open(Path(__file__).parent.joinpath('stopwords_ao3.txt').resolve()) as f:
        for line in f.readlines():
            stopwords.append(line.strip())

    stopwords += stopwordsiso.stopwords('en')

    stopwords += STOP_WORDS

    return stopwords

