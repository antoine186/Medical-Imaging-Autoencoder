from PIL import Image
import numpy as np
import glob, os

# Loads image at specified path as an array.
def image_load(im_path, dtype):

    im = Image.open(im_path)
    im.load()
    form_im = np.asarray(im, dtype=dtype)

    return form_im

# Captures all images found within a file and returning as a cube of images.
def folder_load(folder_path):

    os.chdir(folder_path)
    found_files = glob.glob("*")

    for i in range(len(found_files)):
        cur_im = image_load(folder_path + found_files[i], "float")

        if (i != 0):
            prev_im = np.concatenate((prev_im, cur_im[np.newaxis, :, :, :]), axis = 0)
        elif(i == 0):
            prev_im = cur_im[np.newaxis, :, :, :]

    return prev_im

# Returns all slides for a given type (e.g. adenosis) at a particular magnification
def all_slides_select_type_andmag(type, mag, dim, debug = False):

    type_path_stem_cand1 = "D:/GitHub Projects/Python Based/Neural Network/Datasets/BreaKHis_v1/histology_" \
        "slides/breast/benign/SOB/" + type + "/"

    type_path_stem_cand2 = "D:/GitHub Projects/Python Based/Neural Network/Datasets/BreaKHis_v1/histology_" \
        "slides/breast/malignant/SOB/" + type + "/"

    # This is a trick to make the program find the correct path for a specified cancer type.
    # Bear in mind that those paths only work for my setup. To configure it to yours, modify the above paths.
    if (os.path.exists(type_path_stem_cand1)):
        type_path_stem = type_path_stem_cand1
    else:
        type_path_stem = type_path_stem_cand2

    # Switch directories
    os.chdir(type_path_stem)
    found_directories = glob.glob("*")

    dyn_iter_count = 0

    # Stepping into each folder found within current directory.
    for i in range(len(found_directories)):
        print("Processing directory: " + found_directories[i] + "; " + "Condition: " + type + "; " + "Magnification: "
              + mag)
        cur_im_block = folder_load(type_path_stem + found_directories[i] + "/" + mag + "/")

        # We want to make sure that the images found in our cube abide by our specified dimensions.
        # We can skip because per condition (e.g. adenosis), there are multiple folders containing
        # say mag x40.
        if (debug == True):
            if(cur_im_block.shape[1] != dim[0]):
                print("Skipping this folder as it would cause dimension mismatch")
                continue
            elif(cur_im_block.shape[2] != dim[1]):
                print("Skipping this folder as it would cause dimension mismatch")
                continue

        # This is to differentiate between the first iteration and the second
        if (dyn_iter_count != 0):
            prev_im_block = np.concatenate((prev_im_block, cur_im_block), axis=0)
        elif(dyn_iter_count == 0):
            prev_im_block = cur_im_block

        dyn_iter_count += 1

    return prev_im_block