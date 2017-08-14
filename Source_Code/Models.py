import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, LSTM, MaxPooling2D, Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K

def r2(y_true, y_pred):
    """
    returns the correlation coefficient of y_pred against y_true.

    :param y_true: the true values (independent variable)
    :param y_pred: the predicted values (dependent variable)
    """
    
    SSE = K.sum(K.square(y_true-y_pred))
    SST = K.sum(K.square(y_true-K.mean(y_true)))
    
    return 1-SSE/SST
    
def fullyconnected_multiple_output(activation = 'elu'):
    """
    returns a 9-layer fully connected architecture as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(batch_shape = (None, 60))

    layer1 = Dense(2048, activation=activation)(input_1)
    layer2 = Dense(1024, activation=activation)(layer1)

    layer3 = Dense(512, activation=activation)(layer2)
    layer4 = Dense(256, activation=activation)(layer3)

    layer5 = Dense(128, activation=activation)(layer4)
    layer6 = Dense(64, activation=activation)(layer5)

    layer7 = Dense(32, activation=activation)(layer5)
    last_layer = Dense(16, activation=activation)(layer7)

    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)
    
    feedforward = Model(input = input_1, output=[output1, output2, output3, output4, output5])
    
    feedforward.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2, 'mean_absolute_percentage_error'])
    
    return feedforward

def fullyconnected_single_output(activation = 'elu'):
    """
    returns a 9-layer fully connected architecture as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    feedforward = Sequential()
    
    feedforward.add(Dense(2048, activation=activation, input_dim = 60))
    feedforward.add(Dense(1024, activation=activation))

    feedforward.add(Dense(512, activation=activation))
    feedforward.add(Dense(256, activation=activation))

    feedforward.add(Dense(128, activation=activation))
    feedforward.add(Dense(64, activation=activation))

    feedforward.add(Dense(32, activation=activation))
    feedforward.add(Dense(16, activation=activation))

    feedforward.add(Dense(1))
    
    feedforward.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2, 'mean_absolute_percentage_error'])
    
    return feedforward


def covnet_multiple_8_layers(activation='elu'):
    """
    returns an 8-layer convolutional architecture as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(shape = (40, 50, 1))

    layer1 = Conv2D(32, (12, 12), padding='same', activation=activation)(input_1)
    layer2 = Conv2D(32, (12, 12), padding='same', activation=activation)(layer1)
    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Conv2D(64, (6, 6), padding='same', activation=activation)(layer3)
    layer5 = Conv2D(64, (6, 6), padding='same', activation=activation)(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Conv2D(128, (3, 3), padding='same', activation=activation)(layer6)
    layer8 = Conv2D(128, (3, 3), padding='same', activation=activation)(layer7)
    layer9 = MaxPooling2D(pool_size=(2,2))(layer8)

    flatten = Flatten()(layer9)
    last_layer = Dense(256, activation=activation)(flatten)
    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)

    convnet_mo_elu = Model(input = input_1, output=[output1, output2, output3, output4, output5])

    convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    
    return convnet_mo_elu

def covnet_single_ReLUs_6_layers():
    """
    returns a 6-layer convolutional architecture (with ReLU activation units) as a Keras Model, with outputs for only a single parameter.
    """
    
    convnet = Sequential()
    convnet.add(Conv2D(32, (12, 12), input_shape=(40, 50, 1), padding='same', activation='relu'))
    convnet.add(Conv2D(32, (12, 12), padding='same', activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))

    convnet.add(Conv2D(64, (6, 6), padding='same', activation='relu'))
    convnet.add(Conv2D(64, (6, 6), padding='same', activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))

    convnet.add(Flatten())
    convnet.add(Dense(512, activation='relu'))
    convnet.add(Dense(1))
    
    convnet.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    
    return convnet