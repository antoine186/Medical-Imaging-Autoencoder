
from Path_Crawler import all_slides_select_type_andmag
from sklearn import preprocessing
import numpy as np

# Returns all of the slides of specified magnitude.
# Returns all slides of all conditions concatenated together along the 0 axis.
# Returns a corresponding
def all_slides_select_mag(mag):

    le = preprocessing.LabelEncoder()

    # Benign slides
    adenosis_slides = all_slides_select_type_andmag("adenosis", mag, (460, 700), True)
    fibroadenoma_slides = all_slides_select_type_andmag("fibroadenoma", mag, (460, 700), True)
    phyllodes_tumor_slides = all_slides_select_type_andmag("phyllodes_tumor", mag, (460, 700), True)
    tubular_adenoma_slides = all_slides_select_type_andmag("tubular_adenoma", mag, (460, 700), True)

    # Malignant slides
    ductal_carcinoma_slides = all_slides_select_type_andmag("ductal_carcinoma", mag, (460, 700), True)
    lobular_carcinoma_slides = all_slides_select_type_andmag("lobular_carcinoma", mag, (460, 700), True)
    mucinous_carcinoma_slides = all_slides_select_type_andmag("mucinous_carcinoma", mag, (460, 700), True)
    papillary_carcinoma_slides = all_slides_select_type_andmag("papillary_carcinoma", mag, (460, 700), True)

    all_slides_mag = np.concatenate((adenosis_slides, fibroadenoma_slides, phyllodes_tumor_slides, tubular_adenoma_slides,
                    ductal_carcinoma_slides, lobular_carcinoma_slides, mucinous_carcinoma_slides,
                    papillary_carcinoma_slides), axis=0)

    adenosis_labels = np.repeat("adenosis", adenosis_slides.shape[0])
    fibroadenoma_labels = np.repeat("fibroadenoma", fibroadenoma_slides.shape[0])
    phyllodes_tumor_labels = np.repeat("phyllodes_tumor", phyllodes_tumor_slides.shape[0])
    tubular_adenoma_labels = np.repeat("tubular_adenoma", tubular_adenoma_slides.shape[0])

    ductal_carcinoma_labels = np.repeat("ductal_carcinoma", ductal_carcinoma_slides.shape[0])
    lobular_carcinoma_labels = np.repeat("lobular_carcinoma", lobular_carcinoma_slides.shape[0])
    mucinous_carcinoma_labels = np.repeat("mucinous_carcinoma", mucinous_carcinoma_slides.shape[0])
    papillary_carcinoma_labels = np.repeat("papillary_carcinoma", papillary_carcinoma_slides.shape[0])

    all_slides_mag_labels = np.concatenate((adenosis_labels, fibroadenoma_labels, phyllodes_tumor_labels, tubular_adenoma_labels,
                    ductal_carcinoma_labels, lobular_carcinoma_labels, mucinous_carcinoma_labels,
                    papillary_carcinoma_labels))

    le.fit(all_slides_mag_labels)
    all_slides_mag_labels = le.transform(all_slides_mag_labels)
    #le.inverse_transform(all_slides_mag_labels)

    return all_slides_mag, all_slides_mag_labels, le