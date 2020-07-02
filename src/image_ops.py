"""
def binarize(image):
    #Binarizing image
    arr = image.flatten()
    mean = arr.mean()
    print("mean: {}".format(mean))

"""
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2

def kernel_horizontal(bin_im):
    horizontal = np.copy(bin_im)
    rows = horizontal.shape[1]
    size = rows//40
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    print(structure)
    horizontal = cv2.erode(horizontal, structure)
    horizontal = cv2.dilate(horizontal, structure)
    return horizontal

def conv2d_image(f, w):
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


def otsu_threshold(im):

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

def threshold(pic, threshold):
    return ((pic < threshold) * 255).astype('uint8')

if __name__ == '__main__':

    #fname = 'mysong_0.jpg'
    #image = imageio.imread(fname)
    #img = np.arange(64).reshape(8, 8)
    #binarize(img)

    pic = imageio.imread('../assets/songtron_test_0.png')
    #plt.figure(figsize=(7,7))
    #plt.axis('off')
    #plt.imshow(pic)

    gray = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07])

    bin_img = threshold(gray(pic), otsu_threshold(pic))

    plt.figure(figsize=(7,7))
    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')

    lines = kernel_horizontal(bin_img)
    plt.figure(figsize=(7,7))
    plt.imshow(lines, cmap='gray')
    plt.axis('off')

    bin_img[lines == 255] = 0 

    plt.figure(figsize=(7,7))
    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')

    blurred = cv2.medianBlur(bin_img, 5)
    plt.figure(figsize=(7,7))
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(12,12))
    plt.imshow(closing, cmap='gray')
    plt.axis('off')


    plt.show()


