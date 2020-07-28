from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import numpy as np

def synth_print(nb_fig, original_dim, decoder):
    n = nb_fig
    figure = np.zeros((original_dim[0] * n, original_dim[1] * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded.reshape(original_dim)
            figure[i * original_dim[0]: (i + 1) * original_dim[0],
            j * original_dim[1]: (j + 1) * original_dim[1]] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
