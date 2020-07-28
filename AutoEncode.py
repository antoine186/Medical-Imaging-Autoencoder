from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, MaxPool2D,\
    Flatten, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.metrics import categorical_accuracy
from keras.metrics import binary_crossentropy
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

def autoencode_fit(X, Y, batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch):

    unflat_dim = original_dim
    original_dim = original_dim[0] * original_dim[1]

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # This is the encoder part / Third output is our z latent variable.
    x = Input(shape=(original_dim,), name="input")
    print(x)
    x2 = Reshape((unflat_dim[0], unflat_dim[1], 1), name="x2")(x)
    print(x2)
    h1 = Conv2D(4, kernel_size=(5, 5), activation='relu',
                batch_input_shape=(batch_size, unflat_dim[0], unflat_dim[1], 1), name="h1")(x2)
    print(h1)
    h2 = MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None,
                   name="h2")(h1)
    print(h2)
    h3 = Conv2D(4, kernel_size=(5, 5), activation='relu', name="h3")(h2)
    print(h3)
    h4 = MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None,
                   name="h4")(h3)
    print(h4)
    h5 = Dense(intermediate_dim, activation="relu", name="h5")(h4)
    print(h5)
    h6 = Flatten(name="h6")(h5)
    print(h6)
    """
    z_mean = Dense(latent_dim, name="mean")(h6)
    z_log_var = Dense(latent_dim, name="log-variance")(h6)
    print(z_mean)
    print(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    print(z)
    """
    h7 = Dense(28 * 28, activation="relu", name="h7")(h6)
    print(h7)
    encoder = Model(x, h7, name="encoder")
    print("Model Switch")

    # This is the decoder part / Output is a flattened image.
    input_decoder = Input(shape=(28 * 28,), name="decoder_input")
    print(input_decoder)
    input_decoder2 = Dense(intermediate_dim, activation="relu", name="input_decoder2")(input_decoder)
    print(input_decoder2)
    input_decoder3 = Reshape((16, 16, 1), name="input_decoder3")(input_decoder2)
    print(input_decoder3)
    q1 = Conv2DTranspose(4, kernel_size=(5, 5), activation='relu', name="q1")(input_decoder3)
    print(q1)
    q2 = Conv2DTranspose(4, kernel_size=(5, 5), activation='relu', name="q2")(q1)
    print(q2)
    x_decoded0 = Flatten()(q2)
    x_decoded = Dense(unflat_dim[0] * unflat_dim[1], activation="sigmoid", name="flat_decoded")(x_decoded0)
    print(x_decoded)
    x_decoded2 = Reshape((original_dim,), name="reshaped")(x_decoded)
    print(x_decoded2)
    decoder = Model(input_decoder, x_decoded2, name="decoder")

    #output_combined = decoder(encoder(x)[2])
    output_combined = decoder(encoder(x))
    vae = Model(x, output_combined)
    vae.summary()

    # x_decoded_mean is the reconstruction.
    # This loss function is the combo between binary cross entropy and KL divergence
    """
    def vae_loss(x, x_decoded_mean, original_dim=original_dim):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    """

    vae.compile(optimizer="adam", loss=binary_crossentropy)
    #vae.compile(optimizer="adam", loss=vae_loss)

    # We use the MNIST dataset for debugging purposes
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # We are applying normalisation
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Flattening on a per sample basis.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(x_train.shape)
    print(x_test.shape)

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=nb_epoch,
            batch_size=batch_size,
            validation_data=(x_test, x_test), verbose=1)
            #use_multiprocessing=False

    return encoder, decoder, vae
