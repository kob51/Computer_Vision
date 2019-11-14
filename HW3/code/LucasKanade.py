import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # initialize p
    p = p0
    
    # define the rectangle points on the template image 
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    
    
    #reshape into coordinate matrix of shape 3xN where each column is [x y 1]
    rect = np.reshape(rect,(2,2)).T
    rect = np.concatenate((rect,np.ones((1,2))))
    
    # create an interpolate-able image map of the template (It)
    xcoords_t = np.linspace(0,It.shape[1]-1,It.shape[1])
    ycoords_t = np.linspace(0,It.shape[0]-1,It.shape[0])
    # RBS(x,y,z) --> x is a 2-D array of data with shape (x.size,y.size), so you need
    # to take the transpose (shape is usually in terms of (rows,cols)
    image_bvs_t = RectBivariateSpline(xcoords_t,ycoords_t,It.T)    

    # define all points for which we want to evaluate the warped template.
    # these are all of the points inside the given rectangle
    x_w_t, y_w_t = np.meshgrid(np.linspace(x1,x2+1,int(x2-x1+1)),
                                                 np.linspace(y1,y2+1,int(y2-y1+1)))
    
    #feed thix fxn every coordinate pair in the desired rectangle to get an interpolated 
    #image template
    It = image_bvs_t.ev(x_w_t.flatten(),y_w_t.flatten())
    It = np.reshape(It,(int(y2-y1+1),int(x2-x1+1)))

    # create an interpolate-able image map of the source image (It1)
    xcoords_t1 = np.linspace(0,It1.shape[1]-1,It1.shape[1])
    ycoords_t1 = np.linspace(0,It1.shape[0]-1,It1.shape[0])
    # RBS(x,y,z) --> x is a 2-D array of data with shape (x.size,y.size), so you need
    # to take the transpose (shape is usually in terms of (rows,cols))
    image_bvs_t1 = RectBivariateSpline(xcoords_t1,ycoords_t1,It1.T)
    
    
    tol = .01
    deltap = np.array([1,1])
    while np.linalg.norm(deltap) > tol:
        #build a 2x3 rotation matrix. here we really only want translation,
        # so the matrix will look like
        #                             [[1 0 tx]
        #                              [0 1 ty]]

        warp = np.concatenate((np.eye(2),np.zeros((2,1))),axis=1)
        warp[0,2] = p[0]
        warp[1,2] = p[1]

        #rotate rectangle corner coordinates
        rect_warped = np.matmul(warp,rect)
        x1_w = rect_warped[0,0]
        y1_w = rect_warped[1,0]
        x2_w = rect_warped[0,1]
        y2_w = rect_warped[1,1]
        
        #get all the coordinates inside the warped rectangle
        x_warped, y_warped = np.meshgrid(np.linspace(x1_w,x2_w+1,It.shape[1]),
                                                     np.linspace(y1_w,y2_w+1,It.shape[0]))
        
        #feed thix fxn every coordinate pair in the grid to get the warped
        #rectangle in the source image (It1)
        It1_warped = image_bvs_t1.ev(x_warped.flatten(),y_warped.flatten())
        It1_warped = np.reshape(It1_warped,(It.shape[0],-1))
  
        #compute the error image between the warped version of the source image, 
        #and the template
        error_img = np.subtract(It,It1_warped)
        
        #compute x and y gradients of It1 and combine into an Nx2 matrix [Ix Iy]
        xgrad = np.gradient(It1_warped,axis=1)
        ygrad = np.gradient(It1_warped,axis=0)
        grad = np.vstack((xgrad.flatten(),ygrad.flatten())).T
        
        #Jacobian of warp
        J = np.eye(2)
        
        #compute steepest descent images
        A = np.matmul(grad,J)
#        print(sdi.shape)
        
        #compute Hessian matrix
        H = np.matmul(A.T,A)
        
        #compute steepest descent parameter updates
        sdpi = np.matmul(A.T,error_img.flatten())
        
        # compute the deltap value for this iteration
        deltap = np.matmul(np.linalg.inv(H),sdpi)
        
        # parameter update p <-- p+deltap
        p = np.add(p,deltap)

#    print(p,np.linalg.norm(deltap))
    return p
