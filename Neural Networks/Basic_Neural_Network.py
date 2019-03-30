import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from tensorflow.keras.callbacks import TensorBoard


def CreateModel():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = tf.keras.utils.normalize(x_train, axis=1)
    X_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.save("Basic_Neural_Network.model")

if not(os.path.exists("Basic_Neural_Network.model")):
    CreateModel()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = tf.keras.utils.normalize(x_train, axis=1)
X_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model("Basic_Neural_Network.model")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", mertrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)

print ("{} is this loss ???".format(model.evaluate(X_test, y_test)))
predictions = model.predict([X_test])

name = "mnist_dataset_model_128x2_{}".format(int(time.time()))
tenbor = TensorBoard(log_dir="logs/{}".format(name))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(x_test[i])
# plt.show()
