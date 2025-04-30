import keras
from keras.api.models import Sequential, model_from_json
from keras.api.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, Conv1D, MaxPooling1D


def get_model(x_shape, y_shape, learning_rate, pretrained_weights):
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(x_shape,1)))  # X_train.shape[1] = No. of Columns
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(y_shape)) # Target class number
    model.add(Activation('softmax'))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def get_mode_to_predit(json, weights, learning_rate):
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights)
    print("Loaded model from disk")

    # the optimiser
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return loaded_model