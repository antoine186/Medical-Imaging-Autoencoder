from PIL import Image
import numpy as np
import glob, os

def image_load(im_path, dtype):

    im = Image.open(im_path)
    im.load()
    form_im = np.asarray(im, dtype=dtype)

    return form_im

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

def all_slides_select_type_andmag(type, mag, dim, debug = False):

    type_path_stem_cand1 = "D:/GitHub Projects/Python Based/Neural Network/Datasets/BreaKHis_v1/histology_" \
        "slides/breast/benign/SOB/" + type + "/"

    type_path_stem_cand2 = "D:/GitHub Projects/Python Based/Neural Network/Datasets/BreaKHis_v1/histology_" \
        "slides/breast/malignant/SOB/" + type + "/"

    if (os.path.exists(type_path_stem_cand1)):
        type_path_stem = type_path_stem_cand1
    else:
        type_path_stem = type_path_stem_cand2

    os.chdir(type_path_stem)
    found_directories = glob.glob("*")

    dyn_iter_count = 0

    for i in range(len(found_directories)):
        print("Processing directory: " + found_directories[i] + "; " + "Condition: " + type + "; " + "Magnification: "
              + mag)
        cur_im_block = folder_load(type_path_stem + found_directories[i] + "/" + mag + "/")

        if (debug == True):
            if(cur_im_block.shape[1] != dim[0]):
                print("Skipping this folder as it would cause dimension mismatch")
                continue
            elif(cur_im_block.shape[2] != dim[1]):
                print("Skipping this folder as it would cause dimension mismatch")
                continue

        if (dyn_iter_count != 0):
            prev_im_block = np.concatenate((prev_im_block, cur_im_block), axis=0)
        elif(dyn_iter_count == 0):
            prev_im_block = cur_im_block

        dyn_iter_count += 1

    return prev_im_block