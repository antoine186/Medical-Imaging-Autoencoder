import matplotlib.pyplot as plt
import numpy as np

def test_im(nb1, nb2, decoder, original_dim):
    z = np.array([[1.5, 0.1]])
    raw_pred = decoder.predict(z)
    full_pred = raw_pred.reshape(original_dim)#
    plt.imshow(full_pred, cmap='Greys_r')
    plt.show()