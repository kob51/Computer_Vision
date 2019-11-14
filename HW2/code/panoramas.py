import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH,computeH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    # warp im2 using the homography and make sure the output width is the sum 
    # of both input image widths
    im2_warped = cv2.warpPerspective(im2,H2to1,(im2.shape[1]+im1.shape[1],im2.shape[0]))

    # warp im1 by the identity matrix, but make sure the output width is the
    # sum of both image widths
    pano_im = cv2.warpPerspective(im1,np.eye(3),(im2.shape[1]+im1.shape[1],im1.shape[0]))
    
    # blend the images by overlaying the 2 warped images and taking the maximum
    # value at each element
    pano_im = np.maximum(pano_im,im2_warped)
    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #build a 3x4 array of the corners of im1. each column is [x, y, 1]
    im1_corners = np.ones((3,4))
    im1_corners[:-1,0] = np.array([0, 0])
    im1_corners[:-1,1] = np.array([im1.shape[1]-1,0])
    im1_corners[:-1,2] = np.array([0,im1.shape[0]-1])
    im1_corners[:-1,3] = np.array([im1.shape[1]-1, im1.shape[0]-1])
    
    #build a 3x4 array of the corners of im2. each column is [x, y, 1]
    im2_corners = np.ones((3,4))
    im2_corners[:-1,0] = np.array([0, 0])
    im2_corners[:-1,1] = np.array([im2.shape[1]-1,0])
    im2_corners[:-1,2] = np.array([0,im2.shape[0]-1])
    im2_corners[:-1,3] = np.array([im2.shape[1]-1, im2.shape[0]-1])
    
    
    #transform the im2 coordinates into im1 coordinates and divide each x and y 
    #by its corresponding "z" so that the z for each point is 1
    im2_corners = np.matmul(H2to1, im2_corners)
    im2_corners = np.divide(im2_corners,im2_corners[-1,:]).astype('int')

    # create a 3x8 array of the coordinates of both sets of corners in the im1 frame
    all_corners = np.hstack((im1_corners,im2_corners))
    
    
    # find the minimum and maximum x and y values 
    min_x = np.amin(all_corners[0,:])
    min_y = np.amin(all_corners[1,:])
    
    max_x = np.amax(all_corners[0,:])
    max_y = np.amax(all_corners[1,:])
    
    # create the common homography M that will transform both im1 and im2
    # into a common reference frame such that there will be no clipping
    # in the panorama
    M = np.eye(3)
    
    
    # if min_x is negative, add a translation of abs(min_x) to the M
    # homography. Next, add abs(min_x) to both min_x and max_x so that
    # min_x lies at 0 and max_x is offset accordingly. 
    if min_x < 0:
        M[0,2] = -min_x
        max_x += -min_x
        min_x += -min_x
        
    # if min_y is negative, add a translation of abs(min_y) to the M
    # homograpy. Next, add abs(min_y) to both min_y and max_y so that
    # min_y lies at 0 and max_y is offset accordingly
    if min_y < 0:
        M[1,2] = -min_y
        max_y += -min_y
        min_y += -min_y  
    
    # define the output size as the maximum bounds of the rectangle.
    # max_x and max_y are the maximum COORDINATES, so we need to add 1 to each
    # to get the SHAPE of the image 
    out_size = (int(max_x+1),int(max_y+1))
    
    # warp im1 into the M frame
    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    
    # warp im2 into the M frame via the im1 frame
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    
    
    # blend the images by overlaying the 2 warped images and taking the maximum
    # value at each element
    pano_im = np.maximum(warp_im1, warp_im2)
    
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''
    
    # use BRIEF and RANSAC to find the best homography btwn im1 and im2,
    # and stitch the two images together to make a pretty picture
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H  = ransacH(matches, locs1, locs2)
    np.save('../results/q6_1.npy',H)
    pano_im = imageStitching_noClip(im1, im2, H)
    
    cv2.imwrite('panorama.png', pano_im)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    

    generatePanaroma(im1, im2)
