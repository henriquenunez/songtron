import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2

import time
import sys

class Sheet_music():
    def __init__(self, filename):
        self.filename = filename
        self.bboxes = []
        self.segmentate()

    def kernel_horizontal(self, bin_im):
        horizontal = np.copy(bin_im)
        rows = horizontal.shape[1]
        size = rows//20
        structure = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        horizontal = cv2.erode(horizontal, structure)
        horizontal = cv2.dilate(horizontal, structure)
        return horizontal

    def conv2d_image(self, f, w):
        N, M = f.shape[0:2]
        n, m = w.shape[0:2]
        w_flip = np.flip(np.flip(w, 0), 1)
        a = n//2  # floor of n/2
        b = m//2  # floor of m/2
        g = np.zeros(f.shape, dtype=np.float)
        for x in range(a, N-a):
            for y in range(b, M-b):
                region_f = f[x-a:x+(a+1), y-b:y+(b+1)]
                g[x][y] = np.sum(np.multiply(region_f.astype(np.float), w_flip.astype(np.float)))
        return g

    def otsu_threshold(self, im):
        # Compute histogram and probabilities of each intensity level
        pixel_counts = [np.sum(im == i) for i in range(256)]
        n, m = im.shape[0:2]
        # Initialization
        s_max = (0,0)

        for threshold in range(256):

            # update
            w_0 = sum(pixel_counts[:threshold])
            w_1 = sum(pixel_counts[threshold:])

            mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)])\
                            / w_0 if w_0 > 0 else 0
            mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)])\
                            / w_1 if w_1 > 0 else 0

            # calculate - inter class variance
            s = w_0 * w_1 * (mu_0 - mu_1) ** 2

            if s > s_max[1]:
                s_max = (threshold, s)
        return s_max[0]

    def threshold(self, pic, threshold):
        return ((pic < threshold) * 255).astype('uint8')

    def distance(self, a, b):
        return np.abs(a-b)

    def region_growing_average(self, img, img_t, tolerance, seed, region_n): 
        x = seed[0]; y = seed[1]
        img_t[x, y] = region_n
        avg = np.mean(img[np.where(img_t==region_n)])

        # check matrix border and conquering criterion for the 4-neigbourhood
        if (y+1 < img.shape[1] and img_t[x,y+1] == 0 and self.distance(avg, img[x, y+1]) <= tolerance):
            self.region_growing_average(img, img_t, tolerance, [x, y+1], region_n)

        if (y-1 >= 0 and img_t[x,y-1] == 0  and self.distance(avg, img[x, y-1]) <= tolerance):
            self.region_growing_average(img, img_t, tolerance, [x, y-1], region_n)

        if (x+1 < img.shape[0] and img_t[x+1,y] == 0  and self.distance(avg, img[x+1, y]) <= tolerance):
            self.region_growing_average(img, img_t, tolerance, [x+1, y], region_n)

        if (x-1 >= 0 and img_t[x-1,y] == 0  and self.distance(avg, img[x-1, y]) <= tolerance):
            self.region_growing_average(img, img_t, tolerance, [x-1, y], region_n)

    def do_segmentation(self, image, thresh):
        segment_matrix = np.zeros(image.shape)
        region_counter = 1

        #Will check if region is above some treshold

        qty_pixels = image.shape[0]*image.shape[1]
        pixel_count = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                pixel_count = pixel_count + 1
                progress = int(pixel_count*100/qty_pixels)
                sys.stdout.write("Segmentation [%s%s] %.2f%%\r" % ("#"*progress, "."*(100-progress), (pixel_count*100/qty_pixels)))
                pixel_val = image[x, y]
                if segment_matrix[x, y] == 0 and pixel_val >= thresh: #in case not assigned yet
                    #print("Pixel {} above threshold".format((x,y)))
                    self.region_growing_average(image, segment_matrix, 10, (x, y), region_counter)
                    region_counter += 1
                    #print("X: {} Y: {} Region: {}".format(x,y,region_counter))
        #print("Segmented regions: {}".format(region_counter))
        print()
        return segment_matrix

    def get_bounding_rectangle(self, segment_matrix, segment_number):
        segment_indexes = np.where(segment_matrix == segment_number)
        max_x = segment_indexes[0].max()
        min_x = segment_indexes[0].min()
        max_y = segment_indexes[1].max()
        min_y = segment_indexes[1].min()
        return (min_x, max_x, min_y, max_y)

    def segmentate(self):
        pic = imageio.imread(self.filename)
        gray = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07])

        #---------- Binarize ----------#
        bin_img = self.threshold(gray(pic), self.otsu_threshold(pic))

        #---------- Remove lines ----------#
        lines = self.kernel_horizontal(bin_img)
        bin_img[lines == 255] = 0

        #---------- Morphological closing ----------#
        kernel = np.ones((4,1),np.uint8)
        closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        #---------- Segment notes ----------#
        segmented_notes = self.do_segmentation(closing, self.otsu_threshold(pic))

        mask = np.zeros(segmented_notes.shape)
        mask[segmented_notes > 0] = 1

        #---------- Dilate masks ----------#
        #kernel_dilate = np.ones((3,3),np.uint8)
        #dilated_mask = cv2.dilate(mask, kernel_dilate)

        plt.figure(figsize=(10,2))
        plt.subplot(121)
        plt.title("Original image")
        plt.imshow(pic, cmap='gray')
        plt.axis('off')

        plt.subplot(122)
        plt.title("Dilated mask")
        plt.imshow(closing, cmap='gray')
        plt.axis('off')

        for count, region in enumerate(np.unique(segmented_notes)):
            ranges = self.get_bounding_rectangle(segmented_notes, region)
            #print("Reg. {} Ranges are {}".format(region, ranges))
            temp = np.copy(closing)
            img = temp[ranges[0]:ranges[1],ranges[2]:ranges[3]]
            self.bboxes.append(ranges)

            # Show figures
            #plt.figure()
            #plt.title("Figure nº {}".format(region))
            #plt.imshow(img, cmap='gray')
            #plt.axis('off')

            # Save Images
            #if img.shape[0]>0 and img.shape[1]>0:
            #    filename = '../assets/'+str(count)+'.png'
            #    cv2.imwrite(filename, img)

        print("Found %d bboxes"%(len(self.bboxes)))

        plt.show()
        return self.bboxes

sm = Sheet_music("../assets/dataset4.png")
