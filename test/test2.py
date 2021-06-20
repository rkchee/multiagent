import tensorflow as tf
print(tf.__version__)
import numpy as np
# import tf.keras.backend as K
# from tf.keras import backend as K
# from tf.keras.layers import LSTM, Input
import tensorflow.keras.metrics
import tensorflow.keras.losses

@tf.function
def cool():
    return "ass"

cool()
I = tf.keras.layers.Input(shape=(128, 256, 10)) # unknown timespan, fixed feature size
conv = tf.keras.layers.Conv2D(10, 3, activation='relu', padding='same')(I)
conv = tf.keras.layers.Conv2D(10, 3, activation='relu', padding='same')(conv)
conv = tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same')(conv)
conv = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same')(conv)

# conv = tf.keras.layers.Conv2D(2, 5, strides=(1,1))(conv)
# conv = tf.keras.layers.Conv2D(2, 3)(conv)
# conv = tf.squeeze(conv)
# conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
# conv = tf.keras.layers.Flatten()(conv)
conv = tf.reshape(conv, (1, -1))
f = tf.keras.backend.function(inputs=[I], outputs=[conv])

import numpy as np

data1 = np.random.random(size=(1, 128, 256, 10)) # batch_size = 1, timespan = 100
print(f([data1])[0].shape)
# # (1, 20)

# data2 = np.random.random(size=(1, 314, 200)) # batch_size = 1, timespan = 314
# print(f([data2])[0].shape)