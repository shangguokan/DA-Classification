import os
import json
import numpy as np
import sentencepiece as spm
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def train_and_save_tokenizer(sentences, vocab_size, type, user_defined_symbols, split_by_whitespace, path):
    with open('resource/sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    type_is_word = True if type == 'word' else False

    spm.SentencePieceTrainer.train(
        '--input=resource/sentences.txt --character_coverage=1.0 --bos_id=-1 --eos_id=-1 --pad_id=0 --unk_id=1 --pad_piece=<PAD> --unk_piece=<UNK>' +
        ' --user_defined_symbols='+user_defined_symbols +
        ' --vocab_size='+str(vocab_size) +
        ' --model_type='+type +
        ' --model_prefix='+path.split('.')[0] +
        ' --split_by_whitespace='+str(split_by_whitespace).lower() +
        ' --use_all_vocab='+str(type_is_word).lower()  # https://github.com/google/sentencepiece/issues/200
    )


def load_tokenizer(path='resource/tokenizer.model'):
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


def train_and_save_word2vec(tokenized_sentences, wv_dim, wv_epochs, path):
    with open('resource/tokenized_sentences.txt', 'w') as f:
        for tokenized_sentence in tokenized_sentences:
            f.write(' '.join(tokenized_sentence) + '\n')

    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab(tokenized_sentences)

    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=wv_epochs)
    model.wv.save_word2vec_format(path, binary=True)


def load_word2vec(path, vocabulary, wv_dim, pca_dim):
    vocabulary = list(vocabulary.keys())
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