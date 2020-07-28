import matplotlib.pyplot as plt
import numpy as np

def test_im(decoder, original_dim):
    z = np.random.randint(0.05, 0.95, 256)
    raw_pred = decoder.predict(z)
    full_pred = raw_pred.reshape(original_dim)#
    plt.imshow(full_pred, cmap='Greys_r')
    plt.show()