import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        self.cleff = ()

        # Qty images in reference
        self.reference = {
            "crotchet":2,
            "treble":1,
            "minim": 1,
            "semibrave": 2
        }
        self.notes_tones = []

        self.reference_images = {}
        self.ref_list = []

        self.populate_references()
        self.classify()

    def get_notes(self):
        return self.notes

    def get_cleff(self):
        return self.cleff

    def __feature_mean(self, features_list):
        #Each element in features_list is a tuple
        feats = list(zip(*features_list))

        bw = sum(feats[0])/len(feats[0])
        wh = sum(feats[1])/len(feats[1])
        center_mass_x = sum(feats[2])/len(feats[2])
        center_mass_y = sum(feats[3])/len(feats[3])

        computed = (bw, wh, center_mass_x, center_mass_y)
        print(computed)

        return computed

    def __compute_rms(self, feats):
        rmse_list = []
        #Returns list of computed rmse
        for ref_feat in self.ref_list:
            rmse_list.append(list(map(rmse, list(ref_feat[0]), list(feats))))
        return rmse_list
    """
        #Receives an image, and checks which rmse is the least one.
        def classify(self, img):
            img_feat = extract_features(img)
            print(self.__compute_rms(img_feat))
     """
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
                #print(self.reference_images["treble"][0])
            symbol_match_list = []
            #print("Symbol is: {}".format(symbol))
            for ref_img in self.reference_images[symbol][0]:
                #print("Ref img is {}".format(ref_img))
                symbol_match_list.append(apply_mask(ref_img, img))
            #Will insert the max match and its related symbol
            #print(symbol_match_list)
            general_match_list.append((max(symbol_match_list), symbol))
        print(general_match_list)
        return sorted(general_match_list, key=lambda x : x[0], reverse=True)[0]

    def classify(self):
        notes_found = []
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
                #Classification related computations.
                #First, extract features from image.
                features = extract_features(bbox_img)
                #print(self.__compute_rms(features))
                best_mask = self.__find_best_match_mask(bbox_img)
                #print(best_mask)
                if best_mask[0] > 0.7:
                    #Once we got the best match, here we should weight it to decide which class it belongs to.
                    #For the sake of simplicity, we'll just use the best mask value.
                    #plt.figure()
                    #plt.imshow(bbox_img, cmap='gray')
                    #plt.axis('off')
                    #plt.show()
                    self.notes_tones.append(best_mask[1]) #Get best mask symbol.
                    relative_center = self.get_center(bbox_img)
                    center = [0,0]
                    center[0] = bbox[2] + relative_center[0]
                    center[1] = bbox[0] + relative_center[1]
                    notes_found.append([center, best_mask[1]])
                    if best_mask[1] == 'treble':
                        self.cleff = (bbox[2],bbox[3])

        cleff_found = False
        for note in self.notes:
            if note[1] == 'treble':
                cleff_found = True
        if cleff_found == False:
            print("Cleff not found!")

        # Sort notes
        print("Notes found:")
        print(notes_found)
        sorted_notes = sorted(notes_found, key=lambda x : x[0][0])
        print(sorted_notes)
        for note in sorted_notes:
            #print(note)
            #fig,ax = plt.subplots()
            #detections = self.pic

            #width = 10
            #height = 10
            #rect = patches.Rectangle((note[0][0]-5, note[0][1]-5), height, width, linewidth=1.5, edgecolor='b', facecolor='none')
            #ax.imshow(detections, cmap='gray')
            #ax.add_patch(rect)
            #ax.axis('off')
            #plt.show()

            self.notes.append([note[0][1], note[1]])
        print(self.notes)

    def center_mass(self, img):
        #Okay, first let us compute center of mass of bin image
        white_pixels = np.where(img==255)
        #This gives us two lists, containing the coordinates.
        y_mean = white_pixels[0].sum()/white_pixels[0].size
        x_mean = white_pixels[1].sum()/white_pixels[1].size
        #Since image is already at origin, we can just use height and width.
        return [x_mean, y_mean]

    def get_ellipse(self, bin_im):
        vertical = np.copy(bin_im)
        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(vertical)
        size = vertical.shape[1]
        print(size)
        vertical[:,0:size//4] = 0
        vertical[:,size-(size//4):size] = 0
        #vertical[:][size-4:size-1] = 0
        #disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
        #removing_vertical = cv2.erode(vertical, disk)
        #removing_vertical = cv2.dilate(removing_vertical, disk)
        #plt.subplot(122)
        #plt.imshow(vertical)
        #plt.show()
        return vertical

    def get_center(self, image):
        little_ellipses = self.get_ellipse(image)
        #plt.figure()
        #plt.imshow(little_ellipses, cmap='gray')
        #plt.axis('off')
        return self.center_mass(little_ellipses)

    def get_classified_notes(self):
        return zip(self.notes_tones)
#cn = Classify_notes([],[])

