from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, \
    Dropout, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate
from keras.models import model_from_json, model_from_yaml, load_model, Model, Sequential
from keras.optimizers import SGD

def conv_3_dense_2_global_max_pooling(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(128, activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, activation='relu'))
    model.add(MaxPool2D())
    model.add(GlobalMaxPool2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))


def custom_classifier(architecture):

    pass
