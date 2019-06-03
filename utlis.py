import os
import json
import numpy as np
from gensim.models import Word2Vec
from keras.models import load_model

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_word_vectors(path_to_word2vec, word_frequency):
    word_vector_dim = 300
    vocabulary = list(word_frequency.keys())

    word_vectors = Word2Vec(size=word_vector_dim, min_count=1)
    word_vectors.build_vocab_from_freq(word_frequency)

    word_vectors.intersect_word2vec_format(path_to_word2vec, binary=True)

    # Words without an entry in the binary file are silently initialized to random values.
    # We can detect those vectors via their norms which approach zero.
    words_zero_norm = [word for word in word_vectors.wv.vocab if np.linalg.norm(word_vectors[word]) < 0.05]
    print(' - words not in GoogleNews: %d / %d' % (len(words_zero_norm), len(vocabulary) - 1), words_zero_norm[:50])

    embeddings = np.zeros((len(vocabulary), word_vector_dim))
    # the 0-th row for <TOKEN_PAD>, stays at zero
    # the 1-th and  2-th rows for <PRE_CONTEXT_PAD> and <POST_CONTEXT_PAD> and OOV words are randomly initialized
    for index, word in enumerate(vocabulary[1:], 1):
        embeddings[index] = word_vectors[word]

    # from sklearn.decomposition import PCA
    # embeddings = PCA(n_components=21).fit_transform(embeddings)
    # embeddings[0] = 0

    return embeddings


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def plot_and_save_history(history, path_to_results):
    with open(path_to_results+'training_history.json', 'w') as fp:
        fp.write(json.dumps(history))

    epochs = np.arange(1, len(history['loss'])+1)
    for key in history.keys():
        if key.startswith('val_'):
            k = key.replace('val_', '')

            plt.plot(epochs, history[key])
            plt.title(k)
            plt.ylabel(k)
            plt.xlabel('epoch')

            if k in history.keys():
                plt.plot(epochs, history[k])
                plt.legend(['validation', 'train'])
            else:
                plt.legend(['validation'])

            plt.savefig(path_to_results+k+'.png', dpi=300, bbox_inches='tight')
            plt.clf()


def load_keras_model(path):
    return load_model(
        path,
        custom_objects={}
    )
