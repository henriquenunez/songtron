import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
from utils import *
from feature_extraction import *

class Classify_notes():
    def __init__(self, pic, bboxes):
        # Input
        self.bboxes = bboxes
        self.pic = pic

        # Output
        self.notes = []

        # Qty images in reference
        self.reference = {
            "crotchet":15,
            "treble":1,
            "minim": 12,
            "semibrave": 2
        }
        self.notes_tones = []

        self.reference_images = {}
        self.ref_list = []

        self.populate_references()
        self.classify()

    def __feature_mean(self, features_list):
        #Each element in features_list is a tuple
        feats = list(zip(*features_list))

        bw = sum(feats[0])/len(feats[0])
        wh = sum(feats[1])/len(feats[1])
        mass_spread = sum(feats[2])/len(feats[2])
        #center_mass_y = sum(feats[3])/len(feats[3])

        computed = (bw, wh, mass_spread)#, center_mass_y)
        print(computed)

        return computed

    def __compute_rms(self, feats, symbol):
        symbol_mean_feats = self.reference_images[symbol][1]
        return list(map(rmse, list(symbol_mean_feats), list(feats)))

    def populate_references(self):
        #Loads ref images into memory and exctracts features.
        for symbol in self.reference:
            img_list = [] #Temp list for storing matrices.
            feature_list = [] #Temp list for storing features.
            for i in range(self.reference[symbol]):
                filename = "../assets/reference/"+symbol+"/"+str(i)+".png"
                print(filename)
                img = imageio.imread(filename)
                img_list.append(img)
                feature_list.append(extract_features(img)) #A mean from features is taken.
            self.reference_images[symbol] = (img_list, self.__feature_mean(feature_list))

    #Scans on every category for the best match. Not tinder. Yet.
    def __find_best_match_mask(self, img):
        general_match_list = []
        for symbol in self.reference:
            if symbol == "treble":
                pass
                #print(self.reference_images["treble"][0])
            symbol_match_list = []
            #print("Symbol is: {}".format(symbol))
            for ref_img in self.reference_images[symbol][0]:
                #print("Ref img is {}".format(ref_img))
                symbol_match_list.append(apply_mask(ref_img, img))
            #Will insert the max match and its related symbol
            print(symbol_match_list)
            general_match_list.append((max(symbol_match_list), symbol))
        #print(general_match_list)
        return sorted(general_match_list, key=lambda x : x[0], reverse=True)[0]

    def classify(self):
        #Every bounding box.
        for bbox in self.bboxes:
            #Extract part of image that corresponds to segment being analysed
            bbox_img = self.pic[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            #Just a check if we got a valid region.
            if bbox_img.shape[0]>0 and bbox_img.shape[1]>0:
                """print(self.get_center(bbox_img))
                plt.figure()
                plt.imshow(bbox_img, cmap='gray')
                plt.axis('off')
                plt.show()"""
                
                best_mask = self.__find_best_match_mask(bbox_img)
                #print(best_mask)
                if best_mask[0] > 0.7:
                    #Extract features from image so we can compare
                    features = extract_features(bbox_img)
                    symbol_name = best_mask[1] #name of the class.
                    rmse_feat = self.__compute_rms(features, symbol_name)
                    #If the region is suitable, lets also check out the RMS error.
                    print("Error: {}".format(rmse_feat))
                    if rmse_feat[0] < 0.15 and rmse_feat[2] < 20:
                        print("Imagem passou no crivo de aristóteles!")
                        #Once we got the best match, here we should weight it to decide which class it belongs to.
                        plt.figure()
                        plt.imshow(bbox_img, cmap='gray')
                        plt.axis('off')
                        plt.show()
                        self.notes_tones.append(best_mask[1]) #Get best mask symbol.

    def center_mass(self, img):
        #Okay, first let us compute center of mass of bin image
        white_pixels = np.where(img==0)
        #This gives us two lists, containing the coordinates.
        y_mean = white_pixels[0].sum()/white_pixels[0].size
        x_mean = white_pixels[1].sum()/white_pixels[1].size
        #Since image is already at origin, we can just use height and width.
        return (x_mean, y_mean)

    def get_ellipse(self, bin_im):
        vertical = np.copy(bin_im)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        removing_vertical = cv2.erode(vertical, disk)
        removing_vertical = cv2.dilate(removing_vertical, disk)
        return removing_vertical

    def get_center(self, image):
        little_ellipses = self.get_ellipse(image)
        plt.figure()
        plt.imshow(little_ellipses, cmap='gray')
        plt.axis('off')
        return self.center_mass(little_ellipses)

    def get_classified_notes(self):
        return zip(self.notes_tones)
#cn = Classify_notes([],[])

