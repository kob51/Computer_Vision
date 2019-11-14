import numpy as np
import cv2
import os
from planarH import computeH
from matplotlib import pyplot as plt


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''
    
    
    # ***This method comes from Simon JD Prince Section 15.4***
    
    
    # fist, we eliminate the effect of the intrinsic parameters by
    # pre-multiplying the estimated homography by the inverse of the intrinsic 
    # matrix K. this gives a new homography H_prime
    H_prime = np.matmul(np.linalg.inv(K),H)
    
    # to estimate the first two columns of the rotation matrix R, we compute
    # the SVD of the first two columns of H_prime. then we set L equal to a 3x2
    # zeros matrix with ones along the main diagonal
    U,L,Vt = np.linalg.svd(H_prime[:,:-1])
    L = np.concatenate((np.identity(2),np.zeros((1,2))))


    # the first two columns of R are calculated by multiplying U*L*Vt
    temp = np.matmul(U,L)
    R = np.zeros((3,3))
    R[:,:-1] = np.matmul(temp,Vt)
    
    #the last column of R is found by taking the cross product of the first 2 
    # columns. this guarantees a vector that is also length one and perpendicular 
    # to the first 2 columns
    R[:,-1] = np.cross(R[:,0],R[:,1])
    
    # make sure the sign of R is right by checking to see if the determinant
    # is -1. if it is, multiply the last column by -1
    if np.linalg.det(R) == -1:
        R[:,:-1] = R[:,:-1]*-1
        
    # lambda is the average of the scaling factors between the first 2 columns 
    # of H_prime divided by the first two columns of R. This is equivalent
    # to taking the average of the scaling factors between these 6 elements
    lambda_prime = np.nan_to_num(np.sum(np.divide(H_prime[:,:-1],R[:,:-1])))/6
    
    # t is defined as the third column of the H_prime divided by the scaling
    # factor lambda
    t = (H_prime[:,-1]/lambda_prime).T
    
    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of 3D shape to project
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''
    
    # tack on ones to the 3d points of the input to make them into homogeneous
    # coordinates of the form [x y z 1]
    W = np.vstack((W,np.ones((1,W.shape[1]))))

    # create the rotational/translational matrix by adding the translation
    # as the 4th column of the rotational matrix
    Rt_matrix = np.hstack((R,np.reshape(t,(-1,1))))
    
    # calculate K*Rt_matrix*W and divide all elements by the last row
    # to get the form [x y 1]
    X = np.matmul(np.matmul(K,Rt_matrix), W) 
    X = np.divide(X,X[-1,:])
      
    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    # camera intrinsics
    K = np.array([[3043.72, 0, 1196], [0, 3043.72, 1604],[0, 0, 1]])
    
    # 3d planar coordinates of book corners
    W = np.array([[0, 18.2, 18.2, 0], [0, 0, 26, 26], [0, 0, 0, 0]])
    
    # 2d image coordinates of book corners
    X = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    
    # compute homography to get from the book coordinates in 3d to the
    # book coordinates in 2d. Divide all elements in the homography by the 
    # scale factor H[2,2]
    H_3Dto2D = computeH(X,W[:-1,:])
    H_3Dto2D = H_3Dto2D/H_3Dto2D[2,2]
    
    R, t = compute_extrinsics(K,H_3Dto2D)
    
    # load the [x y z] coordinates of the sphere and make them into homogenous
    # coordinates of the form [x y z 1]
    ballpts = np.loadtxt('../data/sphere.txt')
    ballpts = np.vstack((ballpts,np.ones((1,ballpts.shape[1]))))
    
    # define O_point as the center of the O in "computer" on the textbook cover    
    goal_point = np.array([[826],[1636],[1]])
    
    # multiply by the inverse of H to go from 2d coordinates to 3d coordinates
    # for O_point
    goal_point = np.matmul(np.linalg.inv(H_3Dto2D),goal_point)
    goal_point = np.divide(goal_point,goal_point[-1])

    # define the radius of the sphere (this value was given in the assignment)
    diameter = 6.8581
    radius = diameter/2

    # define a translation matrix for ballpts in 3d so that the ball gets
    # moved to O_point. this matrix is a 4x4 because ballpts is in homogeneous
    # coordinates of form [x y z 1]. the z translation is -radius because we want
    # the bottom of the ball to rest on the O, and the positive z axis comes 
    # OUT of the screen. we want to move the ball back so we need the negative
    # translation INTO the screen
    trans_3d = np.eye(4)
    trans_3d[0,3] = goal_point[0]
    trans_3d[1,3] = goal_point[1]
    trans_3d[2,3] = -radius
    
    # translate the homogenous coordinates of ballpts using trans_3d so
    # that the ball will rest on the O
    ballpts = np.matmul(trans_3d,ballpts)

    # ditch the homogeneous coordinates and project the points onto the 2D
    # image plane
    ballpts = ballpts[:-1]
    ballpts = project_extrinsics(K,ballpts,R,t)

    plt.figure()
    plt.imshow(im)
    plt.scatter(ballpts[0],ballpts[1],color='yellow', s=0.1)
    plt.show()
    
    