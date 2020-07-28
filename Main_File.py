from AutoEncode import autoencode_fit
from Synth_Print import synth_print
from Test_IM import test_im
from tensorflow import keras
from Image_Loader import fast_slides_select_mag
from Image_Loader import all_slides_select_mag
from RGB_2_Gray import r2g
from keras.datasets import mnist
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
# Apply VAE with the mnist dataset
batch_size = 100
original_dim = (28, 28)
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
nb_fig = 15

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test))

encoder, decoder, vae = autoencode_fit(X, Y, batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch)
synth_print(nb_fig, original_dim, decoder)
"""

# Apply VAE with the BreaKHis dataset // This section sometimes experiences exponentially increasing gradient
batch_size = 10
original_dim = (460, 700)
latent_dim = 2
intermediate_dim = 256
nb_epoch = 2
nb_fig = 5

# We are only using one possible condition in order to test our vae
#all_slides_mag, all_slides_mag_labels, le = fast_slides_select_mag("40X")
#all_slides_mag, all_slides_mag_labels, le = all_slides_select_mag("40X")
all_slides_mag = np.load("D:\\GitHub Projects\\Python Based\\Neural Network\\Datasets\\BreaKHis_Cube\\Cube.npy")
all_slides_mag_labels = np.load("D:\\GitHub Projects\\Python Based\\Neural Network\\Datasets\\BreaKHis_Cube\\Labels.npy")

all_slides_mag = all_slides_mag[0:1900,:,:]
all_slides_mag_labels = all_slides_mag_labels[0:1900]

# Balancing samples for batch size / Only when using fast loading
#all_slides_mag = all_slides_mag[0:100,:,:,:]
#all_slides_mag_labels = all_slides_mag_labels[0:100]

"""
# Balancing samples for batch size / Only when using full loading
all_slides_mag = all_slides_mag[0:1970,:,:,:]
all_slides_mag_labels = all_slides_mag_labels[0:1970]

# Converting RGB to grayscale image
all_slides_mag = r2g(all_slides_mag)

# Save/loading gray cube and labels
np.save("D:\\GitHub Projects\\Python Based\\Neural Network\\Datasets\\BreaKHis_Cube\\Cube.npy", all_slides_mag)
np.save("D:\\GitHub Projects\\Python Based\\Neural Network\\Datasets\\BreaKHis_Cube\\Labels.npy", all_slides_mag_labels)
"""

# Fitting our VAE
encoder, decoder, vae = autoencode_fit(all_slides_mag, all_slides_mag_labels, batch_size, original_dim, latent_dim,
                                       intermediate_dim, nb_epoch)

encoder.save("D:\\GitHub Projects\\Python Based\\Neural Network\\AutoEncoder\\encoder")
decoder.save("D:\\GitHub Projects\\Python Based\\Neural Network\\AutoEncoder\\decoder")

encoder = keras.models.load_model("D:\\GitHub Projects\\Python Based\\Neural Network\\AutoEncoder\\encoder")
decoder = keras.models.load_model("D:\\GitHub Projects\\Python Based\\Neural Network\\AutoEncoder\\decoder")

# Displaying a possible subset of synthetic Cancer Images
synth_print(nb_fig, original_dim, decoder)
test_im(decoder, original_dim)