from AutoEncode import autoencode_fit
from Synth_Print import synth_print
from Image_Loader import all_slides_select_mag

batch_size = 100
original_dim = (460, 700)
latent_dim = 2
# About one third
intermediate_dim = 96600
nb_epoch = 5
nb_fig = 15

# Perform command with mnist dataset
#encoder, decoder, vae = autoencode_fit(batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch, True)

#synth_print(nb_fig, original_dim, decoder)

all_slides_mag, all_slides_mag_labels, le = all_slides_select_mag("40X")

encoder, decoder, vae = autoencode_fit(batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch, False, nb_fig)
