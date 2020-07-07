from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

# !!! Plagiarism disclaimer !!!
# A large portion of the code below was taken from the book GANs in Action by Jakub Langr and Vladimir Bok.
# I have reorganised and modified those for my own purpose.

def autoencode_fit(X, Y, batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch, mnist_bool):

    original_dim = original_dim[0] * original_dim[1]

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # This is the encoder part / Third output is our z latent variable.
    x = Input(shape=(original_dim,), name="input")
    h = Dense(intermediate_dim, activation="relu", name="encoding")(x)
    z_mean = Dense(latent_dim, name="mean")(h)
    z_log_var = Dense(latent_dim, name="log-variance")(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")

    # This is the decoder part / Output is a flattened image.
    input_decoder = Input(shape=(latent_dim,), name="decoder_input")
    decoder_h = Dense(intermediate_dim, activation="relu", name="decoder_h")(input_decoder)
    x_decoded = Dense(original_dim, activation="softmax", name="flat_decoded")(decoder_h)
    decoder = Model(input_decoder, x_decoded, name="decoder")

    output_combined = decoder(encoder(x)[2])
    vae = Model(x, output_combined)
    vae.summary()

    # x_decoded_mean is the reconstruction.
    def vae_loss(x, x_decoded_mean, original_dim=original_dim):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    # We use RMSprop, but we could be using Adam or any other stochastic gradient descent-based method
    # Most people stick to RMSprop, SGD, or Adam.
    vae.compile(optimizer="rmsprop", loss=vae_loss)

    # We use the MNIST dataset for debugging purposes
    if mnist_bool:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        #x_train = X[]

        # Changing the range of pixel values.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # Flattening on a per sample basis.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=nb_epoch,
            batch_size=batch_size,
            validation_data=(x_test, x_test), verbose=1)
            #use_multiprocessing=False

    return encoder, decoder, vae
