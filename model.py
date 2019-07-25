import os
import datetime
import numpy as np
import keras.backend as K
from time import time
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Lambda, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from module.S2V import S2V, get_recurrent_layer
from module.TIXIER import TIXIER
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


def train(wv_dim,
          word_vectors,
          fine_tune_word_vectors,
          window_size,
          pre_context_size,
          post_context_size,
          max_seq_len,
          module_name,
          X, Y, SPK,
          epochs, path_to_results):
    n_hidden = wv_dim
    batch_size = 64
    dropout_rate = 0.5

    os.mkdir(path_to_results+'model_on_epoch_end')
    callbacks = [
        ModelCheckpoint(filepath=path_to_results+'model_on_epoch_end/'+'{epoch}.h5'),
        EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=10)
    ]

    # define inputs
    inputs_X = [
        Input(shape=(max_seq_len, ), dtype='int32')
        for _ in range(pre_context_size+window_size+post_context_size)
    ]

    inputs_SPK = [
        Input(shape=(3, ), dtype='float32')
        for _ in range(window_size)
    ]

    # define sentence_encoder
    if module_name == 'TIXIER':
        module = TIXIER(
            pre_context_size=pre_context_size,
            post_context_size=post_context_size,

            input_shape=(max_seq_len,),
            recurrent_name='Bi-GRU',
            pooling_name='attention',
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            path_to_results=path_to_results,
            is_base_network=False,

            with_embdedding_layer=True,
            fine_tune_word_vectors=fine_tune_word_vectors,
            word_vectors=word_vectors,

            with_extra_features=False
        )
    else:
        module = S2V(
            input_shape=(max_seq_len,),
            recurrent_name=module_name,
            pooling_name='attention',
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            path_to_results=path_to_results,
            is_base_network=True,

            with_embdedding_layer=True,
            fine_tune_word_vectors=fine_tune_word_vectors,
            word_vectors=word_vectors,

            with_extra_features=False,
            with_last_f_f_layer=False
        )

    stack_layer = Lambda(K.stack, arguments={'axis': 1})
    recurrent_layer = get_recurrent_layer(
        name='Bi-GRU',
        n_hidden=n_hidden,
        return_sequences=True
    )
    crf_layer = CRF(43, sparse_target=True, unroll=False)

    output = crf_layer(recurrent_layer(stack_layer([
        concatenate([
            inputs_SPK[i-pre_context_size], module(inputs_X[i-pre_context_size:i+post_context_size+1])
        ])
        for i in np.arange(pre_context_size, pre_context_size+window_size, 1)
    ])))

    model = Model(inputs_X + inputs_SPK, output)

    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file=path_to_results+'model.png')
    print(model.summary())

    model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

    training_start_time = time()

    model_trained = model.fit(
        X['train'] + SPK['train'],
        Y['train'],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X['valid']+SPK['valid'], Y['valid']),
        callbacks=callbacks
    )

    print("Training time finished.\n{} epochs in {}".format(epochs, datetime.timedelta(seconds=time() - training_start_time)))

    return model_trained.history
