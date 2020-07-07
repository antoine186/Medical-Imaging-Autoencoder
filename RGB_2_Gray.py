import numpy as np

# We use weighted average to obtain a grayscale image
# This function is supposed to receive a cube
def r2g(im_cube):
    cur_shape = im_cube.shape
    dyn_iter_count = 0

    for i in range(cur_shape[0]):
        if (dyn_iter_count == 0):
            new_cube = np.mean(im_cube[i, :, :, :], axis=2)
            new_cube = new_cube[np.newaxis,:,:]
        else:
            inter_cube = np.mean(im_cube[i, :, :, :], axis=2)
            inter_cube = inter_cube[np.newaxis,:,:]
            new_cube = np.concatenate((new_cube[:,:,:], inter_cube), axis=0)

        dyn_iter_count += 1

    return new_cube
