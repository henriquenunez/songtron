import numpy as np
from math import *
import cv2
from utils import *

"""
Utility functions
"""

#All functions here receive a binarized image.
def bw_ratio(img):
    pixel_count = img.size
    white_pixels = img[img == 255].size
    #print(pixel_count)
    #print(img)
    #print(img[img == 255].size)
    return white_pixels/pixel_count

def wh_ratio(img): #Width over height
    y, x = img.shape
    return x/y

def center_mass_ratio(img):
    height = img.shape[0]
    width = img.shape[1]

    #Okay, first let us compute center of mass of bin image
    white_pixels = np.where(img==0)
    #This gives us two lists, containing the coordinates.
    y_mean = white_pixels[0].sum()/white_pixels[0].size
    x_mean = white_pixels[1].sum()/white_pixels[1].size
    #Since image is already at origin, we can just use height and width.
    return (x_mean/width, y_mean/height)

#will count how many pixels match.
def apply_mask(ref, img):
    #First, resize images to largest dimensions.
    r_height = ref.shape[0]
    r_width = ref.shape[1]
    i_height = ref.shape[0]
    i_width = ref.shape[1]

    if r_height > i_height:
        h = r_height
    else:
        h = i_height
        
    if r_width > i_width:
        w = i_width
    else:
        w = i_width
        
    resized_ref = cv2.resize(ref, (w, h))
    resized_img = cv2.resize(img, (w, h))
    #Time to binarize
    binarized_ref = threshold(resized_ref, otsu_threshold(resized_ref))
    binarized_img = threshold(resized_img, otsu_threshold(resized_img))
 
    print(binarized_ref)
    print(binarized_img)
    count = np.where(resized_ref == resized_img)[0].shape[0]
    return count/(w*h) 

def rmse(a,b):
    return sqrt((a-b)**2)

#This function returns the above features.
def extract_features(img):
    center_mass = center_mass_ratio(img)
    return (bw_ratio(img), wh_ratio(img), center_mass[0], center_mass[1])

class symbol_classifier:
    def __init__(self):
        self.ref_list = []

    def insert_ref(self, features_list, ref_name):
        #Each element in features_list is a tuple
        feats = list(zip(*features_list))
        bw = sum(feats[0])/len(feats[0])
        wh = sum(feats[1])/len(feats[1])
        center_mass_x = sum(feats[2])/len(feats[2])
        center_mass_y = sum(feats[3])/len(feats[3])
        computed = ((bw, wh, center_mass_x, center_mass_y), ref_name)
        print(computed)
        self.ref_list.append(computed)

    def __compute_rms(self, feats):
        rmse_list = []
        #Returns list of computed rmse
        for ref_feat in self.ref_list:
            rmse_list.append(map(rmse, list(ref_feat[0]), list(feats)))
        return rmse_list

    #Receives an image, and checks which rmse is the least one.
    def classify(self, img):
        img_feat = extract_features(img)
        
        print(self.__compute_rms(img_feat))

