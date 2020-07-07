from AutoEncode import autoencode_fit

batch_size = 100
original_dim = 28 * 28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
nb_fig = 15

# Perform command with mnist dataset
autoencode_fit(batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch, True, nb_fig)