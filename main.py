import os
import json
import utlis
import training
import numpy as np
from datetime import datetime
from collections import OrderedDict
from dataset import swda_split, mrda_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from dataset.loader import load_swda_corpus, load_mrda_corpus
from sklearn.metrics import accuracy_score

tokenization_type = 'word'  # bpe, unigram, word
vocab_size = 6000
strip_punctuation = False
split_by_whitespace = True

wv_dim = 64
wv_epochs = 30

swda_conversation_list = swda_split.train_set_idx + swda_split.valid_set_idx + swda_split.test_set_idx
swda_corpus, swda_tag_set, swda_speaker_set, swda_symbol_set = load_swda_corpus(
    swda_conversation_list,
    strip_punctuation=strip_punctuation,
    tokenize_punctuation=True if tokenization_type == 'word' else False
)

mrda_conversation_list = mrda_split.train_set_idx + mrda_split.valid_set_idx + mrda_split.test_set_idx
mrda_corpus, mrda_tag_set, mrda_speaker_set, mrda_symbol_set = load_mrda_corpus(
    mrda_conversation_list,
    strip_punctuation=strip_punctuation,
    tokenize_punctuation=True if tokenization_type == 'word' else False
)

swda_train_val_sentences = [
    sentence for conversation_id in swda_split.train_set_idx + swda_split.valid_set_idx
    for sentence in swda_corpus[conversation_id]['sentence']
]
mrda_train_val_sentences = [
    sentence for conversation_id in mrda_split.train_set_idx + mrda_split.valid_set_idx
    for sentence in mrda_corpus[conversation_id]['sentence']
]
sentences = swda_train_val_sentences + mrda_train_val_sentences
user_defined_symbols = swda_symbol_set.union(mrda_symbol_set)
utlis.train_and_save_tokenizer(
    sentences,
    vocab_size=vocab_size,
    type=tokenization_type,
    user_defined_symbols='▁'+',▁'.join(list(user_defined_symbols)),
    split_by_whitespace=split_by_whitespace,
    path='resource/tokenizer.model'
)
tokenizer = utlis.load_tokenizer('resource/tokenizer.model')
vocabulary = OrderedDict([(tokenizer.id_to_piece(id), id) for id in range(tokenizer.get_piece_size())])

swda_corpus = utlis.tokenize_corpus(swda_corpus, tokenizer)
mrda_corpus = utlis.tokenize_corpus(mrda_corpus, tokenizer)

swda_train_val_tokenized_sentences = [
    sentence for conversation_id in swda_split.train_set_idx + swda_split.valid_set_idx
    for sentence in swda_corpus[conversation_id]['tokenized_sentence']
]
mrda_train_val_tokenized_sentences = [
    sentence for conversation_id in mrda_split.train_set_idx + mrda_split.valid_set_idx
    for sentence in mrda_corpus[conversation_id]['tokenized_sentence']
]
tokenized_sentences = swda_train_val_tokenized_sentences + mrda_train_val_tokenized_sentences
utlis.train_and_save_word2vec(
    tokenized_sentences,
    wv_dim=wv_dim,
    wv_epochs=wv_epochs,
    path='resource/wv_swda.bin'
)
word_embedding_matrix = utlis.load_word2vec('resource/wv_swda.bin', vocabulary, wv_dim=wv_dim, pca_dim=wv_dim)

####################################################
conversation_list = swda_conversation_list
corpus = swda_corpus
tag_set = swda_tag_set
speaker_set = swda_speaker_set
train_set_idx, valid_set_idx, test_set_idx = swda_split.train_set_idx, swda_split.valid_set_idx, swda_split.test_set_idx


seq_lens = [len(seq) for cid in conversation_list for seq in corpus[cid]['sequence']]
maxlen = max(seq_lens)
tag_lb = LabelBinarizer().fit(list(tag_set))
spk_le = LabelEncoder().fit(list(speaker_set))
for cid in conversation_list:
    corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=maxlen, padding='post', truncating='post')
    corpus[cid]['tag'] = tag_lb.transform(corpus[cid]['tag'])
    corpus[cid]['speaker'] = spk_le.transform(corpus[cid]['speaker'])
    corpus[cid]['speaker_change'] = np.not_equal(corpus[cid]['speaker'][:-1], corpus[cid]['speaker'][1:]).astype(int)

X, Y, SPK_C = dict(), dict(), dict()
for key, value in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    X[key] = [corpus[cid]['sequence'] for cid in value]
    Y[key] = [corpus[cid]['tag'] for cid in value]
    SPK_C[key] = [corpus[cid]['speaker_change'] for cid in value]

########################
path_to_results = 'results/' + str(datetime.now()).replace(' ', '_').split('.')[0] + '/'
os.makedirs(path_to_results+'model_on_epoch_end')

crf_type = ['our', 'vanilla'][0]
batch_size = 1
epochs = 50
n_tags = len(tag_lb.classes_)
history, model = training.train(X, Y, SPK_C, word_embedding_matrix, n_tags, epochs, batch_size, crf_type, path_to_results)
utlis.save_and_plot_history(history, path_to_results)

########################
val_loss_list = np.array(history['val_loss'])
val_accuracy_list = np.array(history['val_accuracy'])
best_epoch = int(np.where(val_loss_list == val_loss_list.min())[0][-1] + 1)  # epochs count from 1
val_loss, val_accuracy = val_loss_list[best_epoch-1], val_accuracy_list[best_epoch-1]

print('the best epoch based on val_loss:', best_epoch)
model.load_weights(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')

def viterbi_vanilla_crf(nodes, trans):
    paths = {(k,): v for k, v in nodes[0].items()}
    for l in range(1, len(nodes)):
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items():
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items():
                score = ns + ps + trans[(p[-1], n)]
                if score > max_score:
                    max_path,max_score = p+(n,), score
            paths[max_path] = max_score
    return max(paths, key=paths.get)


def viterbi_our_crf(nodes, trans0, trans1, spk_change_sequence):
    paths = {(k,): v for k, v in nodes[0].items()}
    for l in range(1, len(nodes)):
        paths_old,paths = paths,{}
        trans = trans0 if spk_change_sequence[l-1] == 0 else trans1
        for n,ns in nodes[l].items():
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items():
                score = ns + ps + trans[(p[-1], n)]
                if score > max_score:
                    max_path,max_score = p+(n,), score
            paths[max_path] = max_score
    return max(paths, key=paths.get)

if crf_type == 'vanilla':
    test_loss, test_accuracy = model.evaluate_generator(
        training.data_generator(X, Y, SPK_C, 'test', with_SPK_C=False, batch_size=batch_size), steps=len(X['test']))
    trans = {}

    for i in range(n_tags):
        for j in range(n_tags):
            tag_from = tag_lb.classes_[i]
            tag_to = tag_lb.classes_[j]
            trans[(tag_from, tag_to)] = model.get_layer('vanilla_crf_1').get_weights()[0][i, j]

    y_pred = []
    y_true = []
    n_samples = len(X['test'])
    for i in range(n_samples):
        probas = model.predict(np.array([X['test'][i]]))[0]
        nodes = [dict(zip(tag_lb.classes_, j)) for j in probas]
        tags = viterbi_vanilla_crf(nodes, trans)
        y_pred = y_pred + list(tags)
        y_true = y_true + list(tag_lb.inverse_transform(Y['test'][i]))

    final_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
elif crf_type == 'our':
    test_loss, test_accuracy = model.evaluate_generator(
        training.data_generator(X, Y, SPK_C, 'test', with_SPK_C=True, batch_size=batch_size), steps=len(X['test']))
    trans0, trans1 = {}, {}

    for i in range(n_tags):
        for j in range(n_tags):
            tag_from = tag_lb.classes_[i]
            tag_to = tag_lb.classes_[j]
            trans0[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[0][i, j]
            trans1[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[1][i, j]

    y_pred = []
    y_true = []
    n_samples = len(X['test'])
    for i in range(n_samples):
        probas = model.predict([np.array([X['test'][i]]), np.array([SPK_C['test'][i]])])[0]
        nodes = [dict(zip(tag_lb.classes_, j)) for j in probas]
        tags = viterbi_our_crf(nodes, trans0, trans1, SPK_C['test'][i])
        y_pred = y_pred + list(tags)
        y_true = y_true + list(tag_lb.inverse_transform(Y['test'][i]))

    final_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

print(final_accuracy)

with open(path_to_results + 'result.json', 'w') as f:
    f.write(json.dumps(
        dict(((k, eval(k)) for k in ('crf_type', 'wv_dim', 'vocab_size', 'tokenization_type', 'best_epoch', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy', 'final_accuracy')))
    ))
