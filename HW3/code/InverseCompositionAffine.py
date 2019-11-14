import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    
    p = np.zeros(6)
    
#    # create an interpolate-able image map of the template image (It)
    xcoords_t = np.linspace(0,It.shape[1]-1,It.shape[1])
    ycoords_t = np.linspace(0,It.shape[0]-1,It.shape[0]) 
    
    # get a list of all x coordinates and all y coordinates
    X, Y = np.meshgrid(xcoords_t,ycoords_t)
    X = X.flatten()
    Y = Y.flatten()
#    # create an interpolate-able image map of the source image (It1)
    xcoords_t1 = np.linspace(0,It1.shape[1]-1,It1.shape[1])
    ycoords_t1 = np.linspace(0,It1.shape[0]-1,It1.shape[0])  
#    # RBS(x,y,z) --> x is a 2-D array of data with shape (x.size,y.size), so you need
#    # to take the transpose (shape is usually in terms of (rows,cols))
    image_bvs_t1 = RectBivariateSpline(xcoords_t1,ycoords_t1,It1.T)
    
    #pre-compute x and y gradients of template 
    xgrad = np.gradient(It, axis=1).flatten()
    ygrad = np.gradient(It, axis=0).flatten()
    
    # pre-compute the steepest descent images matrix, and only oncisder the 
    # rows for which the It1_warped points are within the frame of It
    A = np.vstack((xgrad*X,xgrad*Y,xgrad,ygrad*X,ygrad*Y,ygrad)).T
    At_orig = A.T
    
    #pre-compute Hessian matrix
    H = np.matmul(At_orig,A)
    H_inv = np.linalg.inv(H)

    # make a 3xN array of coordinates where each column is [x y 1]
    coords = np.vstack((X,Y,np.ones(X.size)))
    
    deltap = np.array([1,1,1,1,1,1])
    tol = .5

#    M = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    while np.linalg.norm(deltap) > tol:
        #build a 2x3 affine warp matrix that looks like
        #                                              [[1+p1 p2 p3]
        #                                               [p4 1+p5 p6]]
        M = np.array([[1.0+p[0], 0.0+p[1], 0.0+p[2]], 
                      [0.0+p[3], 1.0+p[4], 0.0+p[5]]])
        

        # warp the coordinates with our new M for this iteration
        warped_coords = np.matmul(M,coords)
#
        #get all the coordinates of the warped image
#        print("warping")
        x_warped = warped_coords[0,:]
        y_warped = warped_coords[1,:]
#
        # create a boolean array corresponding to all the points for 
        # which the warped coordinates fit within the template frame
        indices = np.logical_and(x_warped >=0, x_warped < It.shape[1])
        indices = np.logical_and(indices,y_warped >=0)
        indices = np.logical_and(indices,y_warped < It.shape[0])

#        
#        #feed thix fxn every coordinate pair in the grid to get an interpolated 
#        #It1
        It1_warped = image_bvs_t1.ev(x_warped,y_warped)
        It1_warped = np.reshape(It1_warped,(It1.shape))

        #compute the error image between the warped version of the image, and the
        #template. only consider the elements for which the It1_warped points are within
        # the frame of It
        error_img = np.subtract(It,It1_warped).flatten()
        error_img = error_img[indices]

        # sample A.T only at columns that correspond to valid points
        At = At_orig[:,indices]
        
        #compute steepest descent parameter updates
        sdpi = np.matmul(At,error_img)
        
        # compute the deltap value for this iteration
        deltap = np.matmul(H_inv,sdpi)
        
#        print(np.linalg.norm(deltap))
        
        # parameter update p <-- p+deltap
        p = np.add(p,deltap)

#    print(p,np.linalg.norm(deltap))
    
    M = np.vstack((M,np.array([0,0,1])))
    return M
