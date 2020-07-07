from AutoEncode import autoencode_fit
from Synth_Print import synth_print
from Image_Loader import all_slides_select_mag
from Image_Loader import fast_slides_select_mag
from RGB_2_Gray import r2g

batch_size = 10
original_dim = (460, 700)
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
nb_fig = 5

# Perform command with mnist dataset
#encoder, decoder, vae = autoencode_fit(batch_size, original_dim, latent_dim, intermediate_dim, nb_epoch, True)

#synth_print_mnist(nb_fig, original_dim, decoder)

#all_slides_mag, all_slides_mag_labels, le = all_slides_select_mag("40X")
all_slides_mag, all_slides_mag_labels, le = fast_slides_select_mag("40X")

# Balancing samples for batch size
all_slides_mag = all_slides_mag[0:100,:,:,:]
all_slides_mag_labels = all_slides_mag_labels[0:100]

# Converting RGB to grayscale image
all_slides_mag = r2g(all_slides_mag)

encoder, decoder, vae = autoencode_fit(all_slides_mag, all_slides_mag_labels, batch_size, original_dim, latent_dim,
                                       intermediate_dim, nb_epoch, False)

synth_print(nb_fig, original_dim, decoder)
