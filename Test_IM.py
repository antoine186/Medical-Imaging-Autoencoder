from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import random

def test_im(decoder, original_dim):
    z = np.zeros(256)
    for i in range(256):
        z[i] = norm.ppf(random.uniform(0.05, 0.95))
    z = z.tolist()
    raw_pred = decoder.predict([z])
    full_pred = raw_pred.reshape(original_dim)
    plt.imshow(full_pred, cmap='Greys_r')
    plt.show()