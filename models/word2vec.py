import numpy as np

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# using the Google News Word2Vec model (already pretrained and better because we don't have much time)
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

stop_words = set(stopwords.words("english"))

def vector_convert(overview, model):
    words = word_tokenize(overview)