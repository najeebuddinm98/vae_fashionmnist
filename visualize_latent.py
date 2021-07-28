import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress tensorflow warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel(logging.ERROR)  #to suppress tensorflow warnings

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
labels = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
latent_dim=2

encoder = keras.models.load_model('fm_encoder.h5', custom_objects={'latent_dim': latent_dim})
z = encoder.predict(X_train)
#print(z)

fig = plt.scatter(z[:,0], z[:,1], c=y_train, cmap='BrBG')
cbar = plt.colorbar(fig)
cbar.set_ticks(list(range(0,10)))
cbar.set_ticklabels(labels)
plt.show()

