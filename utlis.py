import os
import json
import numpy as np
from gensim.models import Word2Vec

import tensorflow as tf
from keras.models import load_model
from sklearn.decomposition import PCA
from module.LD import Bias
from module.attention_with_vec import AttentionWithVec
from module.attention_with_context import AttentionWithContext
from module.attention_with_time_decay import AttentionWithTimeDecay
from module.TIXIER import Sum
from module.TIXIER import ConcatenateFeatures
from module.TIXIER import ConcatenateContexts
from module.S2V import Max

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_word2vec(path, vocabulary, wv_dim, PCA_dim):
    vocabulary = list(vocabulary.keys())
    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab_from_freq(dict.fromkeys(vocabulary, 1))

    model.intersect_word2vec_format(path, binary=True)

    # Words without an entry in the binary file are silently initialized to random values.
    # We can detect those vectors via their norms which approach zero.
    words_zero_norm = [word for word in model.wv.vocab if np.linalg.norm(model.wv[word]) < 0.05]
    print(' - OOV words: %d / %d' % (len(words_zero_norm), len(vocabulary) - 1), words_zero_norm[:50])

    embeddings = np.zeros((len(vocabulary), wv_dim))
    for index, word in enumerate(vocabulary):
        embeddings[index] = model.wv[word]

    if PCA_dim < wv_dim:
        embeddings = PCA(n_components=PCA_dim).fit_transform(embeddings)

    embeddings[0] = 0
    return embeddings


def train_and_save_word2vec(sentences, wv_dim, path):
    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab(sentences)

    model.train(sentences, total_examples=len(sentences), epochs=30)
    model.wv.save_word2vec_format(path, binary=True)


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def save_and_plot_history(history, path_to_results):
    with open(path_to_results+'history.json', 'w') as f:
        f.write(json.dumps(history))

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


def load_history(path_to_results):
    return json.load(open(path_to_results + 'history.json'))


def load_keras_model(path):
    return load_model(
        path,
        custom_objects={
            'tf': tf,
            'Bias': Bias,
            'AttentionWithVec': AttentionWithVec,
            'AttentionWithContext': AttentionWithContext,
            'AttentionWithTimeDecay': AttentionWithTimeDecay,
            'Sum': Sum,
            'Max': Max,
            'ConcatenateFeatures': ConcatenateFeatures,
            'ConcatenateContexts': ConcatenateContexts
        }
    )