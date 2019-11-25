import numpy as np
import os
import skimage
import skimage.io
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):

    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # multichannel convolution with gaussain filter -- eliminates noise
    bw = skimage.filters.gaussian(image,multichannel=True)

    # change from rgb to grayscale
    bw = skimage.color.rgb2gray(bw)
    
    # get a threshold value and create a binary map based on thresh
    thresh = skimage.filters.threshold_otsu(bw)

    # perform some fancy morphology to clean up the image
    bw = skimage.morphology.closing(bw < thresh, skimage.morphology.square(5))
    
    # label joined regions
    bw = skimage.morphology.label(bw)

    # get data on each region. props is a list of classes describing each
    # labeled reigion
    regions = skimage.measure.regionprops(bw)

    mean_area = 0
    for i in range(len(regions)):
        mean_area += regions[i].bbox_area
    mean_area /= len(regions)

    # only keep bounding boxes that are greater than mean/4.5
    # this eliminates all boxes that are too small to be a letter
    bboxes = [region.bbox for region in regions if region.bbox_area >= mean_area/4.5]
    
    # reformat image to be between 0 and 1 where 0 is writing, 1 is background
    bw[bw == 0] = 255
    bw[np.logical_and(bw > 0,bw < 255)] = 0
    bw[bw==255] = 1

    return bboxes, bw.astype('float')

if __name__ == "__main__":
    image = skimage.io.imread("../images/04_deep.jpg")
    bboxes,bw = findLetters(image)
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',"../images/04_deep.jpg")))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
#        plt.gca().add_patch(rect)
        foo = bw[minr:maxr,minc:maxc]
        plt.imshow(foo)
        plt.show()