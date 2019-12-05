import numpy as np
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from vanilla_crf import VanillaCRF
from our_crf import OurCRF


def data_generator(X, Y, SPK_C, set_name, with_SPK_C, batch_size):
    n_samples = len(X[set_name])
    while True:
        for i in range(n_samples):
            if with_SPK_C:
                yield ([np.array([X['train'][i]]), np.array([SPK_C['train'][i]])],
                       np.array([Y['train'][i]]))
            else:
                yield (np.array([X['train'][i]]),
                       np.array([Y['train'][i]]))


def get_s2v_module(word_embedding_matrix, n_hidden):
    embedding_layer = Embedding(
        word_embedding_matrix.shape[0],
        word_embedding_matrix.shape[1],
        embeddings_initializer=Constant(word_embedding_matrix),
        trainable=False,
        mask_zero=True
    )

    lstm_layer = LSTM(
        units=n_hidden,
        activation='tanh',
        return_sequences=False
    )

    input = Input(shape=(None,), dtype='int32')
    output = lstm_layer(embedding_layer(input))
    model = Model(input, output)
    model.summary()

    return model


def train(X, Y, SPK_C, word_embedding_matrix, n_tags, epochs, batch_size, crf_type, path_to_results):
    n_hidden = word_embedding_matrix.shape[1]

    callbacks = [ModelCheckpoint(filepath=path_to_results+'model_on_epoch_end/'+'{epoch}.h5',
                                 save_weights_only=True),
                 EarlyStopping(monitor='val_loss', patience=10)]

    input_X = Input(shape=(None, None), dtype='int32')

    s2v_module = get_s2v_module(word_embedding_matrix, n_hidden)
    bilstm_layer = Bidirectional(LSTM(units=n_hidden,
                                      activation='tanh',
                                      return_sequences=True),
                                 merge_mode='concat')

    output = bilstm_layer(TimeDistributed(s2v_module)(input_X))

    if crf_type == 'vanilla':
        dense_layer = Dense(n_tags)
        crf = VanillaCRF(ignore_last_label=False)

        output = crf(dense_layer(output))

        model = Model(input_X, output)
        model.summary()
        model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])

        history = model.fit_generator(
            data_generator(X, Y, SPK_C, 'train', with_SPK_C=False, batch_size=batch_size),
            steps_per_epoch=len(X['train']),
            epochs=epochs,
            validation_data=data_generator(X, Y, SPK_C, 'valid', with_SPK_C=False, batch_size=batch_size),
            validation_steps=len(X['valid']),
            callbacks=callbacks
        )
    elif crf_type == 'our':
        input_SPK_C = Input(shape=(None,), dtype='int32')

        dense_layer = Dense(n_tags)
        crf = OurCRF(ignore_last_label=False)

        output = crf(dense_layer(output))

        model = Model([input_X, input_SPK_C], output)
        model.summary()
        model.compile(optimizer='adam', loss=crf.loss_wrapper(input_SPK_C), metrics=[crf.accuracy])

        history = model.fit_generator(
            data_generator(X, Y, SPK_C, 'train', with_SPK_C=True, batch_size=batch_size),
            steps_per_epoch=len(X['train']),
            epochs=epochs,
            validation_data=data_generator(X, Y, SPK_C, 'valid', with_SPK_C=True, batch_size=batch_size),
            validation_steps=len(X['valid']),
            callbacks=callbacks
        )
    else:
        pass

    return history.history, model


