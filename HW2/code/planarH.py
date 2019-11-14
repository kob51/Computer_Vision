import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''
    
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    
    # p1 is x,y
    # p2 is u,v
    
    # we are going FROM u,v TO x,y
    
    x = p1[0,:]
    y = p1[1,:]
    u = p2[0,:]
    v = p2[1,:]
        
    #loop thru the A matrix and put in the values for the system Ah = 0
    A = np.zeros((x.size*2,9))
    for i in range(A.shape[0]):
        if i%2 == 0:
            j = int(i/2)
            A[i,:] = np.array([0, 0, 0, -u[j], -v[j], -1,  y[j]*u[j], y[j]*v[j], y[j]])
        else:
            j = int((i-1)/2)
            A[i,:] = np.array([u[j], v[j], 1, 0, 0, 0, -x[j]*u[j], -x[j]*v[j], -x[j]])
        
    # compute the singular value decomposition and take the row of V_T corresponding
    # to the lowest eigenvalue in S. this will be the best approximation of h
    U, S, V_T = np.linalg.svd(A)

    if S.size < 9:
        H2to1 = V_T[8]
    else:
        H2to1 = V_T[np.argmin(S)]
    
    # reshape h column vector to H matrix
    H2to1 = np.reshape(H2to1,(3,3))
    
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations (2xN)
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    # TO DO ...

    max_inliers = 0
    
    
    # define p1 and p2 as the coordinates of the matching keypoints.
    # both have shape 3xN --> (x,y,1)
    p1 = [locs1[i,:2] for i in matches[:,0]]
    p1 = np.vstack(p1).T
    p1 = np.concatenate((p1,np.ones((1,matches.shape[0]))))
    
    p2 = [locs2[i,:2] for i in matches[:,1]]
    p2 = np.vstack(p2).T
    p2 = np.concatenate((p2,np.ones((1,matches.shape[0]))))
    
    # run the algorithm over the specified number of iterations
    for n in range(num_iter):
        
        # randomly choose 4 sets of matching points
        np.random.shuffle(matches)
        shuffled = matches[:4]

        # define testpoints1 and testpoints2 the same way I defined p1 and p2
        # above
        testpoints1 = [locs1[j,:2] for j in shuffled[:,0]]
        testpoints1 = np.vstack(testpoints1).T

        testpoints2 = [locs2[k,:2] for k in shuffled[:,1]]
        testpoints2 = np.vstack(testpoints2).T
        
        # compute the homography to get from testpoints2 to testpoints1
        H_tmp = computeH(testpoints1, testpoints2)
        
        # multiply all of the matching points p2 by the homography and get 
        # a set of predicted points in the p1 coordinate space
        p1_pred = np.matmul(H_tmp,p2)
        
        # divide all points (x,y,z) in p1_pred by z. during the warping via homography,
        # some scaling was introduced into the points. dividing by that z 
        # value gets us back to (x,y,1) for all points and now we're ready
        # to compare with the actual p1 values
        p1_pred = np.divide(p1_pred,p1_pred[-1,:])
 
        inliers = 0
        # go thru each point in p1_pred and compute the distance to its 
        # expected value in p1. count the number of points for which the distance
        # is less than the predefined tolerance. save the maximum number of
        # inliers and the corresponding H matrix that got you those inliers
        for i in range(matches.shape[0]):
            L2_dist = np.linalg.norm(p1_pred[:,i]-p1[:,i])
            if L2_dist < tol:
                inliers += 1
        
        if inliers > max_inliers:
            max_inliers = inliers
            bestH = H_tmp
        
#print("max inliers",max_inliers)
    
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')#
#    '../data/model_chickenbroth.jpg'
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H = ransacH(matches, locs1, locs2)
    #num_iter=5000