import os
import json
import pandas as pd
import numpy as np
from scipy.special import softmax
import sentencepiece as spm
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, minmax_scale
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

from dataset.loader import tag_name_dict


def train_and_save_tokenizer(sentences, vocab_size, type, user_defined_symbols, split_by_whitespace, path_to_results):
    with open(path_to_results + 'resource/sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    type_is_word = True if type == 'word' else False

    spm.SentencePieceTrainer.train(
        '--input='+path_to_results+'resource/sentences.txt' +
        ' --character_coverage=1.0 --bos_id=-1 --eos_id=-1 --pad_id=0 --unk_id=1 --pad_piece=<PAD> --unk_piece=<UNK>' +
        ' --user_defined_symbols='+user_defined_symbols +
        ' --vocab_size='+str(vocab_size) +
        ' --model_type='+type +
        ' --model_prefix='+path_to_results+'resource/tokenizer' +
        ' --split_by_whitespace='+str(split_by_whitespace).lower() +
        ' --use_all_vocab='+str(type_is_word).lower()  # https://github.com/google/sentencepiece/issues/200
    )


def load_tokenizer(path):
    model = spm.SentencePieceProcessor()
    model.load(path)

    return model


def tokenize_corpus(corpus, tokenizer):
    for cid in corpus.keys():
        corpus[cid]['tokenized_sentence'] = [
            tokenizer.encode_as_pieces(sentence)
            for sentence in corpus[cid]['sentence']
        ]
        corpus[cid]['sequence'] = [
            tokenizer.encode_as_ids(sentence)
            for sentence in corpus[cid]['sentence']
        ]

    return corpus


def train_and_save_word2vec(tokenized_sentences, wv_dim, wv_epochs, path_to_results):
    with open(path_to_results + 'resource/tokenized_sentences.txt', 'w') as f:
        for tokenized_sentence in tokenized_sentences:
            f.write(' '.join(tokenized_sentence) + '\n')

    model = Word2Vec(size=wv_dim)
    model.build_vocab(tokenized_sentences)

    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=wv_epochs)
    model.wv.save_word2vec_format(path_to_results + 'resource/wv.bin', binary=True)

    return list(model.wv.vocab.keys())


def load_word2vec(path, vocabulary, wv_dim, pca_dim, path_to_results):
    with open(path_to_results + 'resource/vocabulary.txt', 'w') as f:
        for word in vocabulary:
            f.write(word + '\n')

    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab_from_freq(dict.fromkeys(vocabulary, 1))

    model.intersect_word2vec_format(path, binary=True)

    # Words without an entry in the binary file are silently initialized to random values.
    # We can detect those vectors via their norms which approach zero.
    words_zero_norm = [word for word in model.wv.vocab if np.linalg.norm(model.wv[word]) < 0.05]
    print(' - words not trained: %d / %d' % (len(words_zero_norm), len(vocabulary) - 1), words_zero_norm)

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


def save_trans_to_csv(weights, header, corpus_name, path_to_results):
    for idx, trans in enumerate(weights):
        if idx == 2:
            break

        df = pd.DataFrame(
            np.exp(trans),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '.csv')

        df = pd.DataFrame(
            minmax_scale(np.exp(trans), axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_scale_tag.csv')
        plt.figure(figsize=(13, 10))
        sns.heatmap(df, xticklabels=1, yticklabels=1, center=1)
        plt.savefig(path_to_results + 'trans' + str(idx) + '_scale_tag.png', dpi=300, bbox_inches='tight')
        plt.clf()

        header_name = [tag_name_dict[corpus_name][t] for t in header]
        df = pd.DataFrame(
            minmax_scale(np.exp(trans), axis=1),
            index=header_name, columns=header_name
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_scale_name.csv')
        plt.figure(figsize=(13, 10))
        sns.heatmap(df, xticklabels=1, yticklabels=1, center=1)
        plt.savefig(path_to_results + 'trans' + str(idx) + '_scale_name.png', dpi=300, bbox_inches='tight')
        plt.clf()

        df = pd.DataFrame(
            normalize(np.exp(trans), norm='l1', axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_standard.csv')

        df = pd.DataFrame(
            softmax(np.exp(trans), axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_softmax.csv')

        df = pd.DataFrame(
            np.argsort(np.argsort(-np.exp(trans), axis=1), axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_sort.csv')


class MyLabelBinarizer(LabelBinarizer):
    """
    https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
    """
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


def encode_as_ids(sentence, word2idx):
    tmp = []
    for word in sentence.split():
        try:
            tmp.append(word2idx[word])
        except:
            tmp.append(1)
    return tmp