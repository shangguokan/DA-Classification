import os
import utlis
import datetime
import keras.backend as K
from time import time
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Lambda, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from module.S2V import S2V
from module.HAN import HAN
from module.LD import LD
from module.TIXIER import TIXIER


def train(wv_dim, fine_tune_word_vectors,
    with_extra_features, module_name,
    epochs, pre_context_size, post_context_size, X, Y,
    max_seq_len, word_vectors, path_to_results):

    context_size = pre_context_size + 1 + post_context_size

    n_hidden = wv_dim
    batch_size = 64
    dropout_rate = 0.5

    os.mkdir(path_to_results+'model_on_epoch_end')
    model_checkpoint = ModelCheckpoint(filepath=path_to_results+'model_on_epoch_end/'+'{epoch}.h5', monitor="val_loss", verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0001, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    callbacks = [model_checkpoint, early_stopping]

    # define inputs
    if with_extra_features:
        inputs = utlis.flatten([
            [Input(shape=(max_seq_len,), dtype='int32'),
             Input(shape=(3,), dtype='float32')]
            for _ in range(context_size)
        ])
    else:
        inputs = [
            Input(shape=(max_seq_len,), dtype='int32')
            for _ in range(context_size)
        ]

    print(inputs)
    # define sentence_encoder
    if module_name == 'LD':
        module = LD(
            context_size=context_size,
            history_sizes=(context_size-1, 0),

            input_shape=(max_seq_len,),
            recurrent_name='LSTM',
            pooling_name='max',
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            path_to_results=path_to_results,
            is_base_network=False,

            with_embdedding_layer=True,
            fine_tune_word_vectors=fine_tune_word_vectors,
            word_vectors=word_vectors,

            with_extra_features=with_extra_features
        )
    elif module_name == 'HAN':
        module = HAN(
            context_size=context_size,

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

            with_extra_features=with_extra_features
        )
    elif module_name == 'TIXIER':
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

            with_extra_features=with_extra_features
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
    gru_layer = GRU(
        units=n_hidden,
        activation='tanh',
        return_sequences=False
    )

    t1 = concatenate([module(inputs[0]), inputs[1]])
    t2 = concatenate([module(inputs[2]), inputs[3]])
    t3 = concatenate([module(inputs[4]), inputs[5]])
    t4 = concatenate([module(inputs[6]), inputs[7]])

    model = Model(inputs, Dense(units=42, activation='softmax')(gru_layer(stack_layer([t1, t2, t3, t4]))))

    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file=path_to_results+'model.png')
    print(model.summary())

    # https://datascience.stackexchange.com/questions/23895/multi-gpu-in-keras
    # https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
    # automatically detect and use *Data parallelism* gpus model
    # try:
    #     model = multi_gpu_model(model, gpus=None)
    # except:
    #     pass

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    training_start_time = time()

    model_trained = model.fit(
        X['train'],
        Y['train'],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X['valid'], Y['valid']),
        callbacks=callbacks
    )

    print("Training time finished.\n{} epochs in {}".format(epochs, datetime.timedelta(seconds=time() - training_start_time)))

    return model_trained.history
