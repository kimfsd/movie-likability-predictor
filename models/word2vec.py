from gensim.models import KeyedVectors

# using the Google News Word2Vec model (already pretrained and better because we don't have much time)
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
