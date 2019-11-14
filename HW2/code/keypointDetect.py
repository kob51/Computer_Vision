import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    DoG_pyramid = []
    
    # take the difference of successive layers of the gaussian pyramid
    for i in range(len(levels)-1):
        DoG_pyramid.append(np.subtract(gaussian_pyramid[:,:,i+1],gaussian_pyramid[:,:,i]))
    
    
    DoG_levels = levels[1:]
    DoG_pyramid = np.dstack(DoG_pyramid)
    
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = list()
    
    
    # loop thru each layer of the DoG pyramid to get the 4 derivatives
    # required for the hessian matrix of each pixel. 
    for i in range(DoG_pyramid.shape[2]):
        hessians = list()
        for h in ["Dxx","Dxy","Dyx","Dyy"]:
            if h == "Dxx":
                temp = cv2.Sobel(DoG_pyramid[:,:,i],ddepth=cv2.CV_64F,dx=2,dy=0)
#                temp = cv2.Sobel(temp,ddepth=cv2.CV_64F,dx=1,dy=0)
#                print(temp)
            elif h == "Dyy":
                temp = cv2.Sobel(DoG_pyramid[:,:,i],ddepth=cv2.CV_64F,dx=0,dy=2)
#                temp = cv2.Sobel(temp,ddepth=cv2.CV_64F,dx=0,dy=1)
            elif h == "Dxy" or h== "Dyx":
                temp = temp = cv2.Sobel(DoG_pyramid[:,:,i],ddepth=cv2.CV_64F,dx=1,dy=1) 
                
#    ***These ones made more sense to me mathematically (correct order of derivatives),
#       but I got better extrema point results using the dx=1,dy=1 method. shrug***   
#            elif h == "Dxy":
#                temp = cv2.Sobel(DoG_pyramid[:,:,i],ddepth=cv2.CV_64F, dx=0,dy=1)
#                temp = cv2.Sobel(temp,ddepth=cv2.CV_64F,dx=1,dy=0)                
#            elif h == "Dyx":
#                temp = cv2.Sobel(DoG_pyramid[:,:,i],ddepth=cv2.CV_64F, dx=1,dy=0)
#                temp = cv2.Sobel(temp,ddepth=cv2.CV_64F,dx=0,dy=1)
                  
            hessians.append(temp)
        
        # compute the curvature ratio R for each pixel in the current pyramid level
        # R = tr(H)^2/det(H)   
        principal_curvature.append(np.nan_to_num(np.divide(np.square(np.add(hessians[0],hessians[3])),
            np.subtract(np.multiply(hessians[0],hessians[3]),np.multiply(hessians[1],hessians[2])))))

    principal_curvature = np.dstack(principal_curvature)
    
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = list()
        
    # compile the locations of all values of DoG_pyramid that fit within
    # the prescribed thresholds. using only these locations will greatly 
    # speed up the compute time
    r_points = abs(principal_curvature) < th_r
    c_points = abs(DoG_pyramid) > th_contrast
    indices = np.logical_and(r_points,c_points)
    indices = np.argwhere(indices)
        
    extrema = 0
    min_index = np.array((0,0,0))
    max_index = np.array(DoG_pyramid.shape)
    for i in range(indices.shape[0]):
        neighborhood= list()
        pxl_pos = indices[i,:]
        
        # loop thru neighboring pixels on the same layer, checking to make
        # sure you are within bounds of the pyramid. if you're within
        # bounds, add the value at test_pos to the neighborhood
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                offset = np.array((dr,dc,0))
                test_pos = np.add(pxl_pos,offset)
                
                if ((test_pos >= min_index).all() and (test_pos < max_index).all()):
                    
                    neighborhood.append(DoG_pyramid[tuple(test_pos)])

        # loop thru the corresponding pixels on the 2 neighboring layers,
        # checking to make sure you are within bounds of the pyramid.
        # if you're within bounds, add the value at test_pos to the neighborhood
        for dl in np.array((-1,1)):
            offset = np.array((0,0,dl))
            test_pos = np.add(pxl_pos,offset)
            
            if ((test_pos >= min_index).all() and (test_pos < max_index).all()):
                
                neighborhood.append(DoG_pyramid[tuple(test_pos)])  
                    
        # if the min or max of the neighborhood is the pixel in question,
        # add it to locs_DoG
        if (min(neighborhood) == DoG_pyramid[tuple(pxl_pos)] or
            max(neighborhood) == DoG_pyramid[tuple(pxl_pos)]):
            
            extrema+=1
            locsDoG.append(pxl_pos)

#    print("extrema found",extrema)
    
    locsDoG = np.vstack(locsDoG)

    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    # run all the functions in this file to get the locations of the keypoints
    # in the image (locs_DoG)
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    PC = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, PC,
        th_contrast, th_r)
    
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    
