import numpy as np
import LucasKanadeAffine as LKA
import scipy.ndimage
import InverseCompositionAffine as ICA

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    M = LKA.LucasKanadeAffine(image1,image2)
#    M = ICA.InverseCompositionAffine(image1,image2)
#    print(M.shape)
#    M = np.vstack((M,np.array([0,0,1])))
#    print(M)
    
    # LK returns the matrix that warps image2 (source) to image1 (template).
    # according to the writeup, we now want to warp image1 to the space of
    # image2. affine_transform takes the INVERSE of the transformation matrix.
    # so the inverse of M(1->2) is M(2->1). so we can just use M here to warp
    # image1 to image2 coordinates
    image1 = scipy.ndimage.affine_transform(image1,M)
    
    # take difference between two images
    diff = np.subtract(image2,image1)
    
    # wherever we see motion above the given tolerance, mark that with a 1
    tol = .25
    mask = np.where(abs(diff) > tol,1,0)
    mask = mask.astype('bool')
    
    mask = scipy.ndimage.morphology.binary_dilation(mask, np.ones((6,6)))
    
    # remove 10 rows and 50 columns from each edge of the mask
    rows = 10
    cols = 50
    mask[:rows,:] = False
    mask[:,:cols] = False
    mask[-rows:,:] = False
    mask[:,-cols:] = False
       
    return mask