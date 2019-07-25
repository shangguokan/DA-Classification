import os
import json
import utlis
import model
import numpy as np
from datetime import datetime
from collections import OrderedDict
from dataset import swda_split, mrda_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from dataset.loader import load_swda_corpus, load_mrda_corpus
from sklearn.metrics import accuracy_score

tokenization_type = 'bpe'  # bpe, unigram, word
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
vocabulary['<PRE_CONTEXT_PAD>'] = len(vocabulary)
vocabulary['<POST_CONTEXT_PAD>'] = len(vocabulary)

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
# utlis.train_and_save_word2vec(
#     tokenized_sentences,
#     wv_dim=wv_dim,
#     wv_epochs=wv_epochs,
#     path='resource/wv_swda.bin'
# )
word_vectors = utlis.load_word2vec('resource/wv_swda.bin', vocabulary, wv_dim=wv_dim, pca_dim=wv_dim)

####################################################
conversation_list = swda_conversation_list
corpus = swda_corpus
tag_set = swda_tag_set
speaker_set = swda_speaker_set
train_set_idx, valid_set_idx, test_set_idx = swda_split.train_set_idx, swda_split.valid_set_idx, swda_split.test_set_idx

window_size = 10
pre_context_size = 3
post_context_size = 2
seq_lens = [len(seq) for cid in conversation_list for seq in corpus[cid]['sequence']]
print('max_seq_len', max(seq_lens))
max_seq_len = 10
padding = 'post'
truncating = 'post'

for cid in conversation_list:
    pad_window_size = window_size - len(corpus[cid]['tag']) % window_size
    if pad_window_size > 0:
        corpus[cid]['tag'].extend(['<TAG_PAD>'] * pad_window_size)
        corpus[cid]['sequence'].extend([[vocabulary['<POST_CONTEXT_PAD>']]] * pad_window_size)
        corpus[cid]['speaker'].extend(['<SPK_PAD>'] * pad_window_size)
    if pre_context_size > 0:
        corpus[cid]['sequence'][:0] = [[vocabulary['<PRE_CONTEXT_PAD>']]] * pre_context_size
        # corpus[cid]['speaker'][:0] = ['<SPK_PAD>'] * pre_context_size
    if post_context_size > 0:
        corpus[cid]['sequence'].extend([[vocabulary['<POST_CONTEXT_PAD>']]] * post_context_size)
        # corpus[cid]['speaker'].extend(['<SPK_PAD>'] * post_context_size)

tag_le = LabelEncoder().fit(list(tag_set)+['<TAG_PAD>'])
spk_lb = LabelBinarizer().fit(list(speaker_set)+['<SPK_PAD>'])
for cid in conversation_list:
    corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=max_seq_len, padding=padding, truncating=truncating)
    corpus[cid]['tag'] = tag_le.transform(corpus[cid]['tag'])
    corpus[cid]['speaker'] = spk_lb.transform(corpus[cid]['speaker'])

for cid in conversation_list:
    corpus[cid]['tag'] = [
        corpus[cid]['tag'][i:i+window_size]
        for i in np.arange(0, len(corpus[cid]['tag']), window_size)
    ]
    corpus[cid]['speaker'] = [
        corpus[cid]['speaker'][i:i+window_size]
        for i in np.arange(0, len(corpus[cid]['speaker']), window_size)
    ]
    corpus[cid]['sequence'] = [
        corpus[cid]['sequence'][i-pre_context_size:i+window_size+post_context_size]
        for i in np.arange(pre_context_size, len(corpus[cid]['sequence'])-post_context_size, window_size)
    ]


X = dict()
Y = dict()
SPK = dict()
for key, value in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    X[key] = np.concatenate([corpus[cid]['sequence'] for cid in value])
    X[key] = list(np.swapaxes(X[key], 0, 1))
    SPK[key] = np.concatenate([corpus[cid]['speaker'] for cid in value])
    SPK[key] = list(np.swapaxes(SPK[key], 0, 1))
    Y[key] = np.concatenate([corpus[cid]['tag'] for cid in value])
    Y[key] = np.expand_dims(Y[key], -1)

########################

module_name = 'TIXIER'
epochs = 50
fine_tune_word_vectors = False
path_to_results = 'results/' + str(datetime.now()).replace(' ', '_').split('.')[0] + '/'
os.mkdir(path_to_results)

history = model.train(wv_dim,
                      word_vectors,
                      fine_tune_word_vectors,
                      window_size,
                      pre_context_size,
                      post_context_size,
                      max_seq_len,
                      module_name,
                      X, Y, SPK,
                      epochs, path_to_results)

utlis.save_and_plot_history(history, path_to_results)

########################

print("load the latest best model based on val_crf_viterbi_accuracy...")
history = utlis.load_history(path_to_results)
val_loss_list = np.array(history['val_loss'])
val_accuracy_list = np.array(history['val_crf_viterbi_accuracy'])

best_epoch = int(np.where(val_accuracy_list == val_accuracy_list.max())[0][-1] + 1)  # epochs count from 1
trained_model = utlis.load_keras_model(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')

val_loss, val_accuracy = val_loss_list[best_epoch-1], val_accuracy_list[best_epoch-1]
test_loss, test_accuracy = trained_model.evaluate(X['test']+SPK['test'], Y['test'])

y_true = np.array(utlis.flatten(utlis.flatten(Y['test'])))
y_pred = np.argmax(utlis.flatten(trained_model.predict(X['test']+SPK['test'])), axis=1)
y_true = tag_le.inverse_transform(y_true)
y_pred = tag_le.inverse_transform(y_pred)

pad_idxs = np.where(y_true == '<TAG_PAD>')[0]
y_true = [tag for i, tag in enumerate(y_true) if i not in pad_idxs]
y_pred = [tag for i, tag in enumerate(y_pred) if i not in pad_idxs]
final_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

with open(path_to_results + 'result.json', 'w') as f:
    f.write(json.dumps(
        dict(((k, eval(k)) for k in ('wv_dim', 'vocab_size', 'tokenization_type', 'pre_context_size', 'post_context_size', 'best_epoch', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy', 'final_accuracy')))
    ))

