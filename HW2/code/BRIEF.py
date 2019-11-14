import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector, createDoGPyramid


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''
    
    # create two arrays that will store the linear indices into the patches
    # that we want to analyze for each keypoint
    
    compareX = np.zeros(nbits,dtype='int')
    compareY = np.zeros(nbits,dtype='int')
    
    # our patch_width is 9 so we want to create indices from 0-81 accounting
    # for every position in the square patch
    for n in range(nbits):
        compareX[n] = int(np.random.randint(0,patch_width**2,1))
        compareY[n] = int(np.random.randint(0,patch_width**2,1))

    return  compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the 
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.
    
    
    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''
    
    
    locs = list()
    desc = list()
    patch_width = int(round(max(np.max(compareX),np.max(compareY))**0.5))
    
    # create an array of [x,y,level] for each keypoint
    for i in range(locsDoG.shape[0]):
        
        # define the temporary BRIEF descriptor for this iteration
        temp_desc = np.zeros(compareX.shape)
        
        # define the center point of the patch at the keypoint
        r = locsDoG[i][0]
        c = locsDoG[i][1]
        radius = patch_width//2
        
        
        # if the patch falls wihin the bounds of the image, perform BRIEF
        # comparisons
        if (r + radius < gaussian_pyramid.shape[0] and r - radius >= 0 and 
            c + radius < gaussian_pyramid.shape[1] and c - radius >= 0):
            
            # define the patch as a square of radius around the keypoint
            patch = gaussian_pyramid[r-radius:r+radius+1,c-radius:c+radius+1,levels[locsDoG[i][2]]]
            
            # compare the values in the patch locations of compareX and compareY.
            # if x is less than y, put a 1 in that index of the descriptor,
            # otherwise leave a zero in there
            for n in range(compareX.size):
                flat_patch = patch.flatten()
                if flat_patch[compareX[n]] < flat_patch[compareY[n]]:
                    temp_desc[n] = 1
            
            
            # only if we've been able to successfully run a BRIEF comparison
            # do we add the given BRIEF descriptor and its corresponding
            # location in the pyramid
            desc.append(temp_desc)
            
            #flip the first two columns of locsDoG because they are in r,c space
            #flipping them gives a coordinate in x,y space
            temp_loc = np.append(np.flip(locsDoG[i,:-1]),levels[locsDoG[i][2]])
            locs.append(temp_loc)
            
            
    locs = np.vstack(locs)
    desc = np.vstack(desc)    
    
    return locs, desc


def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    
    # do all the BRIEF stuff
    test_pattern_file = '../results/testPattern.npy'
    locsDoG, gp = DoGdetector(im)
    DoG, levels = createDoGPyramid(gp)
    compareX, compareY = np.load(test_pattern_file)
    k=0 #unused
    locs,desc = computeBrief(im, DoG, locsDoG, k, levels, compareX, compareY)
    

    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r', linewidth = 0.3)
        plt.plot(x,y,'g.', markersize = 0.6)
    plt.show()    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
    
    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)