import os
import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from gensim.models import Word2Vec
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


def train_and_save_tokenizer(sentences, vocab_size, type='unigram', user_defined_symbols='<PRE_CONTEXT_PAD>,<POST_CONTEXT_PAD>,<CONNECTOR>', path='resource/tokenizer.model'):
    with open('resource/sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write(' '.join(sentence)+'\n')

    spm.SentencePieceTrainer.train(
        '--input=resource/sentences.txt --bos_id=-1 --eos_id=-1 --pad_id=0 --unk_id=1 --pad_piece=<PAD> --unk_piece=<UNK>' +
        ' --user_defined_symbols='+user_defined_symbols +
        ' --vocab_size='+str(vocab_size) +
        ' --model_type='+type +
        ' --model_prefix='+path.split('.')[0]
    )


def load_tokenizer(path='resource/tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def train_and_save_word2vec(sentences, wv_dim, path):
    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab(sentences)

    model.train(sentences, total_examples=len(sentences), epochs=30)
    model.wv.save_word2vec_format(path, binary=True)


def load_word2vec(path, vocabulary, wv_dim, pca_dim):
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

    if pca_dim < wv_dim:
        embeddings = PCA(n_components=pca_dim).fit_transform(embeddings)

    embeddings[0] = 0
    return embeddings


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