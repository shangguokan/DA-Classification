import os
import json
import utlis
import model
import numpy as np
from datetime import datetime
from dataset.swda_split import *
from dataset.loader import load_swda_corpus
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences

conversation_list = train_set_idx + valid_set_idx + test_set_idx
corpus, vocabulary, tag_set, speaker_set = load_swda_corpus(conversation_list, concatenate_interruption=True)

sentences = [sent for cid in train_set_idx + valid_set_idx for sent in corpus[cid]['text']]
utlis.train_and_save_word2vec(sentences, wv_dim=64, path='resource/wv_swda.bin')

word_vectors = utlis.load_word2vec('resource/wv_swda.bin', vocabulary, wv_dim=64, PCA_dim=64)
# word_vectors = utlis.load_word2vec('resource/GoogleNews-vectors-negative300-SLIM.bin.gz', vocabulary, wv_dim=300, PCA_dim=64)

pre_context_size = 3
post_context_size = 0
context_size = pre_context_size + 1 + post_context_size

seq_lens = []
for cid in conversation_list:
    for seq in corpus[cid]['sequence']:
        seq_lens.append(len(seq))

    if pre_context_size > 0:
        corpus[cid]['text'][:0] = [['<PRE_CONTEXT_PAD>']] * pre_context_size
        corpus[cid]['sequence'][:0] = [[vocabulary['<PRE_CONTEXT_PAD>']]] * pre_context_size
        corpus[cid]['speaker'][:0] = ['-'] * pre_context_size

    if post_context_size > 0:
        corpus[cid]['text'].extend([['<POST_CONTEXT_PAD>']] * post_context_size)
        corpus[cid]['sequence'].extend([[vocabulary['<POST_CONTEXT_PAD>']]] * post_context_size)
        corpus[cid]['speaker'].extend(['-'] * post_context_size)
max_seq_len = 50  # max(seq_lens)

tag_lb = LabelBinarizer().fit(list(tag_set))
spk_lb = LabelBinarizer().fit(list(speaker_set)+['-'])
for cid in conversation_list:
    corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=max_seq_len, padding='post', truncating='post')
    corpus[cid]['tag'] = tag_lb.transform(corpus[cid]['tag'])
    corpus[cid]['speaker'] = spk_lb.transform(corpus[cid]['speaker'])

    if context_size > 0:
        n_sample = len(corpus[cid]['tag'])
        corpus[cid]['sequence'] = [corpus[cid]['sequence'][i:i+n_sample] for i in range(context_size)]
        corpus[cid]['speaker'] = [corpus[cid]['speaker'][i:i+n_sample] for i in range(context_size)]

X = dict()
Y = dict()
for key, conversation_list in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    list1 = list(np.concatenate([corpus[cid]['sequence'] for cid in conversation_list], axis=1))
    list2 = list(np.concatenate([corpus[cid]['speaker'] for cid in conversation_list], axis=1))
    X[key] = [None] * (len(list1) + len(list2))
    X[key][::2] = list1
    X[key][1::2] = list2
    Y[key] = np.concatenate([corpus[cid]['tag'] for cid in conversation_list], axis=0)

########################

word_vectors_name = 'Word2Vec'
fine_tune_word_vectors = True
with_extra_features = True
module_name = 'TIXIER'
epochs = 50
path_to_results = 'results/' + str(datetime.now()).replace(' ', '_').split('.')[0] + '/'
os.mkdir(path_to_results)

history = model.train(
    word_vectors_name, fine_tune_word_vectors,
    with_extra_features, module_name,
    epochs, pre_context_size, post_context_size, X, Y,
    max_seq_len, word_vectors, path_to_results
)

utlis.plot_and_save_history(history, path_to_results)

####################
print("load the latest best model based on val_categorical_accuracy...")
# Load *the latest best model* according to the quantity monitored (val_categorical_accuracy)
val_categorical_accuracies = np.array(json.load(open(path_to_results+'training_history.json'))['val_categorical_accuracy'])
best_epoch = np.where(val_categorical_accuracies == val_categorical_accuracies.max())[0][-1] + 1  # epochs count from 1
print(" - best epoch: " + str(best_epoch))
model = utlis.load_keras_model(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')

print(model.evaluate(X['test'], Y['test']))
