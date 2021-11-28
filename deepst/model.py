import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPool3D,
    Dropout,
    Flatten,
    Activation,
    Add,
    Dense,
    Reshape,
    BatchNormalization
)
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape
)
# import keras
# from keras.layers.convolutional import Convolution2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from tensorflow.keras.layers import Conv3D, Layer
# from tensorflow.keras.engine.topology import Layer
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
class iLayer(Layer):
    '''
    final weighted sum
    '''
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

# custom layer for branches fusion
class LinearLayer(tf.keras.layers.Layer):
  def __init__(self, name1=None):
      # pass
    # self.name1 = name1
    super(LinearLayer, self).__init__()

    # self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel1 = self.add_weight("kernel1", shape = (int(input_shape[0][1]), int(input_shape[0][2]), int(input_shape[0][3]), int(input_shape[0][4]))) # kernel1 shape []
    self.kernel2 = self.add_weight("kernel2", shape = (int(input_shape[1][1]), int(input_shape[1][2]), int(input_shape[1][3]), int(input_shape[1][4])))
    self.kernel3 = self.add_weight("kernel3", shape = (int(input_shape[2][1]), int(input_shape[2][2]), int(input_shape[2][3]), int(input_shape[2][4])))
    # self.kernel3 = self.add_weight(name=self.name1, shape = (int(input_shape[1]), int(input_shape[2]), int(input_shape[3]), int(input_shape[4])))
    super(LinearLayer, self).build(input_shape)


  def call(self, inputs):
    # return tf.math.multiply(inputs, self.kernel3)
    return (
        tf.math.multiply(inputs[0], self.kernel1)
        + tf.math.multiply(inputs[1], self.kernel2)
        + tf.math.multiply(inputs[2], self.kernel3)
    )

class iLayer(Layer):
    '''
    final weighted sum
    '''
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape


'''
    lmst3d_resnet implementation for BikeNYC and TaxiNYC
'''
def lmst3d_resnet_nyc(len_c, len_p, len_t, nb_flow=2, map_height=16, map_width=8, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for len in [len_c, len_p, len_t]:
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)

            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            kernel_size = (2,3,3)

            conv1 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(input)
            dropout1 = Dropout(0.25)(conv1)
            print(dropout1.shape)

            # the second layers have 64 filters
            nb_filters = 64




            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            dropout2 = Dropout(0.25)(conv2)
            print(dropout2.shape)

            outputs.append(dropout2)

            conv3 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout2)
            dropout3 = Dropout(0.25)(conv3)
            print(dropout3.shape)



            outputs= keras.layers.Add()([input, dropout3])

            outputs.append(dropout3)



    # parameter-matrix-based fusion
    # if len(outputs) == 0 :
    #     main_output = outputs[0]
    # else:
    #     # from .iLayer import iLayer
    # new_outputs = []
    # for output in outputs:
    #     new_outputs.append(iLayer()(output))
    # fusion = Add()(new_outputs)

    # parameter-matrix-based fusion
    # fusion = Add()([LinearLayer("k1")(outputs[0]), LinearLayer("k2")(outputs[1]), LinearLayer("k3")(outputs[2])])
    # fusion = Add()([Dense(64)(outputs[0]), Dense(64)(outputs[1]), Dense(64)(outputs[2])])
    fusion = LinearLayer()(outputs)
    # temp = fusion(outputs)
    # flatten = Reshape((-1, 2048))(fusion)

    # new_outputs = []
    # for output in outputs:
    #     new_outputs.append(iLayer()(output))
    # fusion = keras.layers.Add()(new_outputs)
    # fusion = Activation('relu')(fusion)

    print(fusion.shape)
    # model.add(Flatten())
    flatten = Flatten()(fusion)
    # flatten = Dense(2048)(fusion)
    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(nb_filters * 2 * map_height/4 * map_width/4)(embedding)
        h1 = Dense(flatten.shape[1])(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model
