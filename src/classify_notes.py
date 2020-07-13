import matplotlib.pyplot as plt
import cv2
import imageio

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
                plt.figure()
                plt.imshow(bbox_img, cmap='gray')
                plt.axis('off')
        plt.show()

#cn = Classify_notes([],[])
