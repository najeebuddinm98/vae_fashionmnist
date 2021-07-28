import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress tensorflow warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel(logging.ERROR)  #to suppress tensorflow warnings

decoder = keras.models.load_model('fm_decoder.h5', compile=False)

#Set the parameters of the plot
n = 20 #20x20 grid
figure = np.zeros((28 * n, 28 * n, 1))
#Set the limits of the area to be explored within our latent space
grid_x = np.linspace(1, 3, n)
grid_y = np.linspace(0, 2, n)[::-1]

#Decode an image for each vector
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded.reshape((28,28,1))
        figure[i*28:(i+1)*28, j*28:(j+1)*28] = digit


#Plotting
plt.figure(figsize=(10, 10))
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1]))
plt.imshow(figure, cmap='gnuplot2')
plt.show()  
