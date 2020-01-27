import os
import json
import string
import secrets
import numpy as np
from math import ceil
from datetime import datetime
from keras import backend as K
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from keras.preprocessing.sequence import pad_sequences
import utlis
import trainer
from utlis import MyLabelBinarizer
from dataset import swda_split, mrda_split
from dataset.loader import load_swda_corpus, load_mrda_corpus


param_grid = {
    'tokenization_type': ['word'],
    'vocab_size_bpe_unigram': [6000],
    'strip_punctuation': [False],

    'wv_dim': [300],
    'wv_epochs': [20],

    'corpus_name': ['swda', 'mrda'],
    'swda_concatenate_interruption': [True],
    'mrda_tag_map': ['basic'],  # basic, general, full
    'encoder_type': ['lstm', 'bilstm', 'att-bilstm'],
    'mode': ['vanilla_crf', 'vanilla_crf-spk', 'our_crf-spk_c'],
    'batch_size': [1],
    'dropout_rate': [0.2],
}

for param in ParameterGrid(param_grid):
    tokenization_type = param['tokenization_type']  # word, bpe, unigram
    vocab_size_bpe_unigram = param['vocab_size_bpe_unigram']
    strip_punctuation = param['strip_punctuation']

    wv_dim = param['wv_dim']
    wv_epochs = param['wv_epochs']

    corpus_name = param['corpus_name']
    swda_concatenate_interruption = param['swda_concatenate_interruption']
    mrda_tag_map = param['mrda_tag_map']
    encoder_type = param['encoder_type']
    mode = param['mode']
    batch_size = param['batch_size']
    dropout_rate = param['dropout_rate']

    ###################################
    fid = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    path_to_results = 'results/' + str(datetime.now()).replace(' ', '_').split('.')[0] + '_' + fid + '/'
    os.makedirs(path_to_results + 'model_on_epoch_end')
    os.makedirs(path_to_results + 'resource')

    swda_conversation_list = swda_split.train_set_idx + swda_split.valid_set_idx + swda_split.test_set_idx
    swda_corpus, swda_tag_set, swda_speaker_set, swda_symbol_set = load_swda_corpus(
        swda_conversation_list,
        strip_punctuation=strip_punctuation,
        tokenize_punctuation=True if tokenization_type == 'word' else False,
        concatenate_interruption=swda_concatenate_interruption
    )

    mrda_conversation_list = mrda_split.train_set_idx + mrda_split.valid_set_idx + mrda_split.test_set_idx
    mrda_corpus, mrda_tag_set, mrda_speaker_set, mrda_symbol_set = load_mrda_corpus(
        mrda_conversation_list,
        strip_punctuation=strip_punctuation,
        tokenize_punctuation=True if tokenization_type == 'word' else False,
        tag_map=mrda_tag_map
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
        vocab_size=vocab_size_bpe_unigram,
        type=tokenization_type,
        user_defined_symbols='▁'+',▁'.join(list(user_defined_symbols)),
        split_by_whitespace=True,
        path_to_results=path_to_results
    )
    tokenizer = utlis.load_tokenizer(path_to_results + 'resource/tokenizer.model')
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
        path_to_results=path_to_results
    )
    word_embedding_matrix = utlis.load_word2vec(path_to_results + 'resource/wv_swda.bin', vocabulary, wv_dim=wv_dim, pca_dim=wv_dim)

    ####################################################
    if corpus_name == 'swda':
        conversation_list = swda_conversation_list
        corpus = swda_corpus
        tag_set = swda_tag_set
        speaker_set = swda_speaker_set
        train_set_idx, valid_set_idx, test_set_idx = swda_split.train_set_idx, swda_split.valid_set_idx, swda_split.test_set_idx
    else:
        conversation_list = mrda_conversation_list
        corpus = mrda_corpus
        tag_set = mrda_tag_set
        speaker_set = mrda_speaker_set
        train_set_idx, valid_set_idx, test_set_idx = mrda_split.train_set_idx, mrda_split.valid_set_idx, mrda_split.test_set_idx


    seq_lens = [len(seq) for cid in conversation_list for seq in corpus[cid]['sequence']]
    tag_lb = MyLabelBinarizer().fit(list(tag_set))
    spk_le = LabelEncoder().fit(list(speaker_set))
    spk_lb = MyLabelBinarizer().fit(range(len(speaker_set)))
    for cid in conversation_list:
        corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=max(seq_lens), padding='post', truncating='post')
        corpus[cid]['tag'] = tag_lb.transform(corpus[cid]['tag'])
        corpus[cid]['speaker'] = spk_le.transform(corpus[cid]['speaker'])
        corpus[cid]['speaker_change'] = np.not_equal(corpus[cid]['speaker'][:-1], corpus[cid]['speaker'][1:]).astype(int)
        corpus[cid]['speaker'] = spk_lb.transform(corpus[cid]['speaker'])

    X, Y, SPK, SPK_C = dict(), dict(), dict(), dict()
    for key, value in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
        X[key] = [corpus[cid]['sequence'] for cid in value]
        Y[key] = [corpus[cid]['tag'] for cid in value]
        SPK[key] = [corpus[cid]['speaker'] for cid in value]
        SPK_C[key] = [corpus[cid]['speaker_change'] for cid in value]

    ########################

    epochs = 100
    n_tags = len(tag_lb.classes_)
    n_spks = len(spk_lb.classes_)
    history, model = trainer.train(X, Y, SPK, SPK_C, encoder_type, word_embedding_matrix, n_tags, n_spks, epochs, batch_size, dropout_rate, mode, path_to_results)
    utlis.save_and_plot_history(history, path_to_results)

    ########################
    val_loss_list = np.array(history['val_loss'])
    best_epoch = int(np.where(val_loss_list == val_loss_list.min())[0][-1] + 1)  # epochs count from 1
    val_loss = val_loss_list[best_epoch-1]

    print('the best epoch based on val_loss:', best_epoch)
    model.load_weights(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')
    n_test_samples = len(X['test'])
    test_loss = model.evaluate_generator(
        trainer.data_generator('test', X, Y, SPK, SPK_C, mode, batch_size),
        steps=ceil(n_test_samples/batch_size))


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


    if mode == 'vanilla_crf' or mode == 'vanilla_crf-spk':
        trans = {}
        for i in range(n_tags):
            for j in range(n_tags):
                tag_from = tag_lb.classes_[i]
                tag_to = tag_lb.classes_[j]
                trans[(tag_from, tag_to)] = model.get_layer('vanilla_crf_1').get_weights()[0][i, j]
        utlis.save_trans_to_csv(model.get_layer('vanilla_crf_1').get_weights(), tag_lb.classes_, corpus_name, path_to_results)

        y_pred = []
        y_true = []
        for i in range(n_test_samples):
            if mode == 'vanilla_crf':
                probas = model.predict(np.array([X['test'][i]]))[0]
            if mode == 'vanilla_crf-spk':
                probas = model.predict([np.array([X['test'][i]]), np.array([SPK['test'][i]])])[0]
            nodes = [dict(zip(tag_lb.classes_, j)) for j in probas[:, :n_tags]]
            tags = viterbi_vanilla_crf(nodes, trans)
            y_pred = y_pred + list(tags)
            y_true = y_true + list(tag_lb.inverse_transform(Y['test'][i]))

        final_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

    if mode == 'our_crf-spk_c':
        trans0, trans1 = {}, {}
        for i in range(n_tags):
            for j in range(n_tags):
                tag_from = tag_lb.classes_[i]
                tag_to = tag_lb.classes_[j]
                trans0[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[0][i, j]
                trans1[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[1][i, j]
        utlis.save_trans_to_csv(model.get_layer('our_crf_1').get_weights(), tag_lb.classes_, corpus_name, path_to_results)

        y_pred = []
        y_true = []
        for i in range(n_test_samples):
            probas = model.predict([np.array([X['test'][i]]), np.array([SPK_C['test'][i]])])[0]
            nodes = [dict(zip(tag_lb.classes_, j)) for j in probas[:, :n_tags]]
            tags = viterbi_our_crf(nodes, trans0, trans1, SPK_C['test'][i])
            y_pred = y_pred + list(tags)
            y_true = y_true + list(tag_lb.inverse_transform(Y['test'][i]))

        final_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

    print(final_accuracy)

    with open(path_to_results + 'result.json', 'w') as f:
        f.write(json.dumps(
            dict(((k, eval(k)) for k in ('tokenization_type', 'vocab_size_bpe_unigram', 'strip_punctuation', 'wv_dim', 'wv_epochs', 'corpus_name', 'swda_concatenate_interruption', 'mrda_tag_map', 'encoder_type', 'mode', 'batch_size', 'dropout_rate', 'best_epoch', 'val_loss', 'test_loss', 'final_accuracy')))
        ))
    K.clear_session()
