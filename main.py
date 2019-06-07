import os
import re
import json
import nltk
import utlis
import model
import numpy as np
from datetime import datetime
from dataset.swda_split import *
from dataset.swda.swda import CorpusReader
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict, OrderedDict
from keras.preprocessing.sequence import pad_sequences

corpus = CorpusReader('dataset/swda/swda')
UTT = defaultdict(list)
TAG = defaultdict(list)
SPK = defaultdict(list)
DAMSL_tags = set()

vocabulary = OrderedDict({'<TOKEN_PAD>': 0, '<PRE_CONTEXT_PAD>': 1, '<POST_CONTEXT_PAD>': 2})
frequency = OrderedDict({'<TOKEN_PAD>': 0, '<PRE_CONTEXT_PAD>': 1, '<POST_CONTEXT_PAD>': 1})

pre_context_size = 3
post_context_size = 0
context_size = pre_context_size + 1 + post_context_size

for trans in corpus.iter_transcripts():
    conversation_id = 'sw' + str(trans.conversation_no)
    if conversation_id not in train_set_idx + valid_set_idx + test_set_idx:
        continue

    if pre_context_size > 0:
        UTT[conversation_id].extend([[vocabulary['<PRE_CONTEXT_PAD>']]]*pre_context_size)
        SPK[conversation_id].extend(['-']*pre_context_size)

    for utt in trans.utterances:
        words = ' '.join(utt.text_words(filter_disfluency=True))
        # print(words)
        words = re.sub(r'<<.*>>', '', words)
        words = re.sub(r'\*.*$', '', words)
        words = re.sub(r'[#)(]', '', words)
        words = re.sub(r'--', '', words)
        words = re.sub(r' -$', '', words)
        while True:
            output = re.sub(r'([A-Z]) ([A-Z])\b', '\\1\\2', words)
            if output == words:
                break
            words = output
        tokenizer = nltk.tokenize.TweetTokenizer()
        words = tokenizer.tokenize(words)
        words = ' '.join(words)
        words = re.sub(r' -s', 's', words)
        words = re.sub(r' -', '-', words)
        # print(words)

        words = words.split()
        if len(words) == 0 or words == ['.']:
            continue

        # words to indexes
        w2i = []
        for word in words:
            if word not in vocabulary.keys():
                word_idx = len(vocabulary.keys())
                vocabulary[word] = word_idx
                w2i.append(word_idx)
                frequency[word] = 1
            else:
                w2i.append(vocabulary[word])
                frequency[word] += 1

        utt_tag = utt.damsl_act_tag()
        if utt_tag != '+':
            UTT[conversation_id].append(w2i)
            TAG[conversation_id].append(utt_tag)
            SPK[conversation_id].append(utt.caller)
            DAMSL_tags.add(utt_tag)
        # resolve issue related to tag '+' (check section 2 of the Coders' Manual)
        else:
            concatenated = False
            for i in reversed(range(len(UTT[conversation_id]))):
                if SPK[conversation_id][i] == utt.caller:
                    UTT[conversation_id][i].extend(w2i)
                    concatenated = True
                    break
            # swda.py#195 should be commented out
            assert concatenated is True

    if post_context_size > 0:
        UTT[conversation_id].extend([[vocabulary['<POST_CONTEXT_PAD>']]]*post_context_size)
        SPK[conversation_id].extend(['-']*post_context_size)

seq_lens = []
for conversation_id in train_set_idx + valid_set_idx + test_set_idx:
    for utt in UTT[conversation_id]:
        seq_lens.append(len(utt))
max_seq_len = 50  # max(seq_lens)

tag_lb = LabelBinarizer().fit(list(DAMSL_tags))
spk_lb = LabelBinarizer().fit(['A', 'B', '-'])
for conversation_id in train_set_idx + valid_set_idx + test_set_idx:
    UTT[conversation_id] = pad_sequences(UTT[conversation_id], maxlen=max_seq_len, padding='post', truncating='post')
    TAG[conversation_id] = tag_lb.transform(TAG[conversation_id])
    SPK[conversation_id] = spk_lb.transform(SPK[conversation_id])

    if context_size > 0:
        n_sample = len(TAG[conversation_id])
        UTT[conversation_id] = [UTT[conversation_id][i:i+n_sample] for i in range(context_size)]
        SPK[conversation_id] = [SPK[conversation_id][i:i+n_sample] for i in range(context_size)]

X = dict()
Y = dict()
for key, conversation_list in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    list1 = list(np.concatenate([UTT[conversation_id] for conversation_id in conversation_list], axis=1))
    list2 = list(np.concatenate([SPK[conversation_id] for conversation_id in conversation_list], axis=1))
    X[key] = [None] * (len(list1) + len(list2))
    X[key][::2] = list1
    X[key][1::2] = list2
    Y[key] = np.concatenate([TAG[conversation_id] for conversation_id in conversation_list], axis=0)

########################

word_vectors_name = 'News'
word_vectors = utlis.load_word_vectors('resource/GoogleNews-vectors-negative300-SLIM.bin.gz', frequency)
fine_tune_word_vectors = False
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
