import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np

import classifier

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
            "treble":1
        }
        self.reference_images = {}

        self.populate_reference_images()
        self.classify()

    def populate_reference_images(self):
        '''
            Load all reference images to the memory
        '''
        for symbol in self.reference:
            for i in range(self.reference[symbol]):
                filename = "../assets/reference/"+symbol+"/"+str(i)+".png"
                print(filename)
                img = imageio.imread(filename)
                if i == 0:
                    self.reference_images[symbol] = []
                self.reference_images[symbol].append(img)

    def classify(self):
        for bbox in self.bboxes:
            bbox_img = self.pic[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            if bbox_img.shape[0]>0 and bbox_img.shape[1]>0:
                print(self.get_center(bbox_img))
                plt.figure()
                plt.imshow(bbox_img, cmap='gray')
                plt.axis('off')
                plt.show()

    def center_mass_ratio(self, img):

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
        return self.center_mass_ratio(little_ellipses)
#cn = Classify_notes([],[])
