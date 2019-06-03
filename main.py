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

corpus = CorpusReader('dataset/swda/swda/swda')
UTT = defaultdict(list)
TAG = defaultdict(list)
SPK = defaultdict(list)

vocabulary = OrderedDict({'<TOKEN_PAD>': 0, '<PRE_CONTEXT_PAD>': 1, '<POST_CONTEXT_PAD>': 2})
frequency = OrderedDict({'<TOKEN_PAD>': 0, '<PRE_CONTEXT_PAD>': 1, '<POST_CONTEXT_PAD>': 1})

DAMSL_tags = set()
max_seq_len = 0

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

    # swda.py#195 should be commented out, to include any utterance with act_tag ended with a '@' (e.g. 'b@') in trans.utterances,
    # even though @ marked slash-units with bad segmentation, they are important for understanding the conversation
    # check section 1c of the Coders' Manual http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
    for utt_idx, utt in enumerate(trans.utterances):
        utt_tag = utt.damsl_act_tag()
        # resolve issue related to tag '+'
        # check section 2 of the Coders' Manual:
        if utt_tag == '+':
            for i in reversed(range(utt_idx)):
                if trans.utterances[i].caller == utt.caller and trans.utterances[i].damsl_act_tag() != '+':
                    utt_tag = trans.utterances[i].damsl_act_tag()
                    break
        DAMSL_tags.add(utt_tag)

        if utt_tag == 'x':
            # print(utt.text)
            words = re.sub(r'[#/]', '', utt.text).strip()
            words = [(str(s) + '>').strip() for s in words.split('>') if s]
            words = ' '.join(words)
            # print(words)
        else:
            words = ' '.join(utt.text_words(filter_disfluency=True))
            # print(words)
            words = re.sub(r' -$', '', words)
            words = re.sub(r' --$', '', words)
            words = re.sub(r'^-- ', '', words)
            words = re.sub(r'[#()]', '', words)
            tokenizer = nltk.tokenize.TweetTokenizer()
            words = tokenizer.tokenize(words)
            words = ' '.join(words)
            words = re.sub(r' -', '-', words)
            words = re.sub(r'< < ', '<<', words)
            words = re.sub(r' > >', '>>', words)
            # print(words)
        words = words.split()

        if len(words) == 0:
            continue

        if len(words) > max_seq_len:
            max_seq_len = len(words)

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

        UTT[conversation_id].append(w2i)
        TAG[conversation_id].append(utt_tag)
        SPK[conversation_id].append(utt.caller)

    if post_context_size > 0:
        UTT[conversation_id].extend([[vocabulary['<POST_CONTEXT_PAD>']]]*post_context_size)
        SPK[conversation_id].extend(['-']*post_context_size)

assert len(DAMSL_tags) == 42

max_seq_len = 10
tag_lb = LabelBinarizer().fit(list(DAMSL_tags))
spk_lb = LabelBinarizer().fit(['A', 'B', '-'])
for conversation_id in train_set_idx + valid_set_idx + test_set_idx:
    UTT[conversation_id] = pad_sequences(UTT[conversation_id], maxlen=max_seq_len, padding='post')
    TAG[conversation_id] = tag_lb.transform(TAG[conversation_id])
    SPK[conversation_id] = spk_lb.transform(SPK[conversation_id])

    if context_size > 0:
        n_sample = len(TAG[conversation_id])
        UTT[conversation_id] = [UTT[conversation_id][i:i+n_sample] for i in range(context_size)]
        SPK[conversation_id] = [SPK[conversation_id][i:i+n_sample] for i in range(context_size)]


X = dict()
Y = dict()

for key, conversation_list in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    X[key] = list(np.concatenate([UTT[conversation_id] for conversation_id in conversation_list], axis=1))
    Y[key] = np.concatenate([TAG[conversation_id] for conversation_id in conversation_list], axis=0)

########################

word_vectors_name = 'News'
word_vectors = utlis.load_word_vectors('resource/GoogleNews-vectors-negative300-SLIM.bin.gz', frequency)
fine_tune_word_vectors = False
with_extra_features = False
module_name = 'TIXIER'
epochs = 10
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
print("load the latest best model based on val_loss...")
# Load *the latest best model* according to the quantity monitored (val_loss)
val_losses = np.array(json.load(open(path_to_results+'training_history.json'))['val_loss'])
best_epoch = np.where(val_losses == val_losses.min())[0][-1] + 1  # epochs count from 1
print(" - best epoch: " + str(best_epoch))
model = utlis.load_keras_model(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')

model.evaluate(X['test'], Y['test'])
