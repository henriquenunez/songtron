import numpy as np
from math import *
import cv2
from utils import *

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

#This function returns the above features.
def extract_features(img):
    center_mass = center_mass_ratio(img)
    return (bw_ratio(img), wh_ratio(img), center_mass[0], center_mass[1])

