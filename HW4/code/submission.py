"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import cv2
import helper as H
import scipy.ndimage


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # scale the data by dividing all coordinate by M (maximum of image's
    # height and width)
    T = np.array([[1./M, 0, 0],[0, 1./M, 0],[0, 0, 1]])
    pts1 = np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2 = np.hstack((pts2,np.ones((pts2.shape[0],1))))  
    pts1 = np.matmul(pts1,T)
    pts2 = np.matmul(pts2,T)

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    
    # create our A matrix. this is created by simplifying the equation
    # x2^T * F * x1 = 0 and reformatting into A*f = 0 where f is a column
    # vector containing all 9 entries of the fundamental matrix F
    
    # x1 and x2 are of the form [x y 1]^T
    A = np.zeros((pts1.shape[0],9))
    A[:,0] = x1*x2
    A[:,1] = x2*y1
    A[:,2] = x2
    A[:,3] = x1*y2
    A[:,4] = y1*y2
    A[:,5] = y2
    A[:,6] = x1
    A[:,7] = y1
    A[:,8] = 1
    
    # the entries of F correspond to the last row of V_T (last column of V)
    U,S,V_T = np.linalg.svd(A)
    F = V_T[-1,:]
    F = np.reshape(F,(3,3))

    # refine the result using local minimization, and setting the last
    # singular value to 0, enforcing that rank(F) = 2
    pts1 = pts1[:,:-1]
    pts2 = pts2[:,:-1]
    F = H.refineF(F,pts1,pts2)
    
    # get unscaled coordinates by evaluating T^T * F * T
    F = np.matmul(T.T,np.matmul(F,T))

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # create scaling matrix T
    T = np.array([[1./M, 0, 0],[0, 1./M, 0],[0, 0, 1]])

    # turn pts1 and pts2 into homogeneous coordinates and scale by T
    pts1 = np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2 = np.hstack((pts2,np.ones((pts2.shape[0],1))))  
    pts1 = np.matmul(pts1,T)
    pts2 = np.matmul(pts2,T)

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    
    # create our A matrix. this is created by simplifying the equation
    # x2^T * F * x1 = 0 and reformatting into A*f = 0 where f is a column
    # vector containing all 9 entries of the fundamental matrix F
    A = np.zeros((pts1.shape[0],9))
    A[:,0] = x1*x2
    A[:,1] = x2*y1
    A[:,2] = x2
    A[:,3] = x1*y2
    A[:,4] = y1*y2
    A[:,5] = y2
    A[:,6] = x1
    A[:,7] = y1
    A[:,8] = 1

    U,S,V_T = np.linalg.svd(A)
    
    #F1 and F2 are the last 2 rows of V_T
    F1 = V_T[-1,:]
    F1 = np.reshape(F1,(3,3))
    F2 = V_T[-2,:]
    F2 = np.reshape(F2,(3,3))
    
    # define coefficients of our polynomial
    fun = lambda x: np.linalg.det(x * F1 + (1 - x) * F2)
    a_0 = fun(0)
    a_1 = 2 * (fun(1) - fun(-1))/3 - (fun(2) - fun(-2))/12
    a_2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
    a_3 = fun(1) - a_0 - a_1 - a_2

    # get the roots of the polynomial and filter out the imaginary ones
    roots = np.roots((a_3,a_2,a_1,a_0))
    roots = roots[np.isreal(roots)]

    # get all possible F matrices
    Farray = [a*F1+(1-a)*F2 for a in roots]

    # refine F's
    # pts1 = pts1[:,:-1]
    # pts2 = pts2[:,:-1]
    # Farray = [H.refineF(F, pts1, pts2) for F in Farray]
    
    # denormalize F
    Farray = [np.matmul(T.T, np.matmul(F, T)) for F in Farray]
    return Farray
        
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    
    # E = K2^T * F * K1
    E = np.matmul(K2.T,np.matmul(F,K1))
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    
    x1 = pts1[:,0]
    y1 = pts1[:,1]

    x2 = pts2[:,0]
    y2 = pts2[:,1]
    
    # create A matrix (described in more detail in writeup)
    A1 = np.vstack((C1[2,0]*x1 - C1[0,0], C1[2,1]*x1 - C1[0,1], C1[2,2]*x1 - C1[0,2], C1[2,3]*x1 - C1[0,3])).T
    A2 = np.vstack((C1[2,0]*y1 - C1[1,0], C1[2,1]*y1 - C1[1,1], C1[2,2]*y1 - C1[1,2], C1[2,3]*y1 - C1[1,3])).T
    A3 = np.vstack((C2[2,0]*x2 - C2[0,0], C2[2,1]*x2 - C2[0,1], C2[2,2]*x2 - C2[0,2], C2[2,3]*x2 - C2[0,3])).T
    A4 = np.vstack((C2[2,0]*y2 - C2[1,0], C2[2,1]*y2 - C2[1,1], C2[2,2]*y2 - C2[1,2], C2[2,3]*y2 - C2[1,3])).T
    
    P = list()
    for i in range(pts1.shape[0]):
        # grab rows of A matrix needed for i-th point
        A = np.vstack((A1[i],A2[i],A3[i],A4[i]))
        
        # 3D point in homogeneous coordinates is the last row of V_T
        U,S,V_T = np.linalg.svd(A)
        p_temp = V_T[-1,:]
        p_temp = p_temp / p_temp[-1]
        P.append(p_temp)
        
    P = np.vstack(P)
    
    error = 0
    for i in range(pts1.shape[0]):
        # calculate projected 2D points by multiplying 3D point by camera matrix
        proj1 = np.matmul(C1,P[i,:])
        proj1 = proj1[0:2]/proj1[-1]
        
        proj2 = np.matmul(C2,P[i,:])
        proj2 = proj2[0:2]/proj2[-1]
        
        #calculate reprojection error (played with all these error metrics, not sure
        # which one is right, but they're all similar..?)
        temp_error = np.linalg.norm((proj1-pts1[i]))**2 + np.linalg.norm((proj2-pts2[i]))**2
        # temp_error = np.sum(np.abs(proj1-pts1[i]))**2 + np.sum(np.abs(proj2-pts2[i]))**2
#        temp_error = np.sum(np.subtract(proj1,pts1[i]))**2 + np.sum(np.subtract(proj2,pts2[i]))**2
#        temp_error = np.sum((proj1-pts1[i])**2 + (proj2-pts2[i])**2)
        error += temp_error
        
    # delete last column of P, making it of the form N X 3 (x, y, z)
    P = P[:,:-1]

    return P, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    
    # window radius --> 5 pixels in each direction from pixel
    win_rad = int(5)
    
    # since the images are very similar, we only want to search over a subset
    # of points along the epipolar line
    search = 20
    
    # get vector representing epipolar line on im2
    # l2 = [a b c] --> ax+by+c=0
    pt = np.vstack((x1,y1,1))
    l2 = np.matmul(F,pt)
    a = l2[0]
    b = l2[1]
    c = l2[2]
    
    # define some y points and solve for the correpsonding x points along
    # the epipolar line in im2
    y2pts = np.arange(y1-search,y1+search).astype('int')
    x2pts = np.round((-c-b*y2pts)/a).astype('int')
    
    h = im2.shape[0]
    w = im2.shape[1]
    
    # filter out all points for which the window will be beyond the bounds
    # of the image
    xvalid = np.logical_and(x2pts >= win_rad, x2pts + win_rad < w)
    yvalid = np.logical_and(y2pts >= win_rad, y2pts + win_rad < h)
    valid = np.logical_and(xvalid,yvalid)
    x2pts = x2pts[valid]
    y2pts = y2pts[valid]
    
    # create the windwo
    win1 = im1[y1-win_rad:y1+win_rad+1,x1-win_rad:x1+win_rad+1]
    
    # gaussian filter? works pretty well without this...
#    win1 = scipy.ndimage.gaussian_filter(win1,5)
    
    # find the window in im2 for which the distance between the window in im1 
    # is minimized
    min_dist = -1
    for i in range(x2pts.size):
        x2_t = x2pts[i]
        y2_t = y2pts[i]
        win2 = im2[y2_t-win_rad:y2_t+win_rad+1,x2_t-win_rad:x2_t+win_rad+1]
        
        # gaussian filter? works pretty well without this...
#        win2 = scipy.ndimage.gaussian_filter(win2,5)
        
        dist = np.sum((win1-win2)**2)
        
        if dist < min_dist or min_dist == -1:
            min_dist = dist
            x2 = x2_t
            y2 = y2_t
            
    return x2,y2

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    
    # predefine pts1 and pts2 in homogeneous coordinates
    pts1_h = np.hstack((pts1,np.ones((pts1.shape[0],1)))).T
    pts2_h = np.hstack((pts2,np.ones((pts2.shape[0],1)))).T

    pts = np.hstack((pts1,pts2))

    num_inliers = -1
    iter = 1000
    tol = 1
    for i in range(iter):
        # randomly get 7 corresponding points
        pts_test = np.random.permutation(pts)[:7]
        pts1_test = pts_test[:,0:2]
        pts2_test = pts_test[:,2:]
        
        # calculate list of possible fundamental matrices
        Flist = sevenpoint(pts1_test,pts2_test,M)
        
        for Ftest in Flist:
            # calculate the epipolar lines on im1 and im2
            l1_array = np.matmul(Ftest.T,pts2_h)
            l2_array = np.matmul(Ftest,pts1_h)

            a1 = l1_array[0]
            b1 = l1_array[1]
            c1 = l1_array[2]
            a2 = l2_array[0]
            b2 = l2_array[1]
            c2 = l2_array[2]
            
            # calculate distances btwn pts on im1 and epipolar lines on im1
            dist1 = (a1*x1 + b1*y1 + c1)/np.sqrt(a1**2 + b1**2)
            # calculate distances btwn pts on im2 and epipolar lines on im2
            dist2 = (a2*x2 + b2*y2 + c2)/np.sqrt(a2**2 + b2**2)
            # distance metric is sum of these 2 quantities squared.
            # sometimes this would produce imaginary values of 0j?
            # not sure why but i suppressed the imaginary parts just to be safe
            dist = np.real(dist1**2+dist2**2)

            # inliers are points for which the distance metric is less than a tolerance
            test_inliers = dist < tol

            # update variables if we have more inliers than num_inliers
            if np.sum(test_inliers) > num_inliers:
                num_inliers = np.sum(test_inliers)
                inliers = test_inliers
                F = Ftest

    # refine F using the final list of inliers
    F = H.refineF(F,pts1[inliers],pts2[inliers])

    return F,inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # takes in a vector r where r specifies the axis, and the norm of the r
    # specifies theta, the angle you want to rotate around the axis
    theta = np.sqrt(np.sum(r**2))
    u = r/theta
    I = np.eye(3)
    if theta == 0:
        R = I
    else:
        u = r/theta
        u_cross = np.array([[0, -u[2,0], u[1,0]], [u[2,0], 0, -u[0,0]], [-u[1,0], u[0,0], 0]])
        R = I*np.cos(theta) + (1-np.cos(theta))*u*u.T + u_cross * np.sin(theta)
    return R
'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
     theta = np.arccos((np.trace(R)-1)/2)
     if theta != 0:
         omega = 1.0 / (2*np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]],
                                                         [R[0, 2] - R[2, 0]],
                                                         [R[1, 0] - R[0, 1]]])
         r = omega * theta
     return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # unpack x. last 6 elements are r and t vectors
    # rest of elements are flattened 3D points
    P = x[:-6]
    r2 = x[-6:-3]
    t2 = x[-3:]

    num_pts = p1.shape[0]

    # reshape P into a list of Nx3, corresponding to 3D points
    P = np.reshape(P,(num_pts,3))

    # compute camera matrix for camera 1
    C1 = np.matmul(K1,M1)

    # compute the camera matrix for camera 2 
    r2 = np.reshape(r2, (3, 1))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2))
    C2 = np.matmul(K2,M2)

    # convert 3D points to homogeneous coordinates for matrix mutliplication
    # with C's (3x4 matrices)
    P_h = np.vstack((P.T, np.ones((1, num_pts))))

    # compute projections of P's on im1
    p1_hat = np.matmul(C1,P_h)
    p1_hat = (p1_hat[0:2]/p1_hat[2]).T

    # compute projections of P's on im2
    p2_hat = np.matmul(C2,P_h)
    p2_hat = (p2_hat[0:2]/p2_hat[2]).T

    # compute the resisduals, which is the error we will try to minimize in 
    # bundle adjustment. It represents the distance between the reprojected
    # points and the actual points in each image.
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # build the x vector (structure described in rodriguesResidual)
    R2_init = M2_init[:,0:3]
    t2_init = M2_init[:,3].flatten()
    r2_init = invRodrigues(R2_init).flatten()
    P_init = P_init.flatten()
    x_init = np.hstack((P_init,r2_init,t2_init))

    # least squares to get the value of x corresponding to minimal residual.
    # the leastsq function produced very inconsistent results, but 
    # the minimize from piazza function produced consistently good results. in theory they
    # should both do the same thing? not sure what the problem is

    # residuals = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    # x_final, _ = scipy.optimize.leastsq(residuals,x_init)
    func = lambda x: (rodriguesResidual(K1,M1,p1,K2,p2,x)**2).sum()
    x_final = scipy.optimize.minimize(func,x_init,method='L-BFGS-B').x

    # get final 3D points
    P2 = x_final[:-6]
    P2 = np.reshape(P2,(-1,3))

    # get final extrinsics matrix M2
    r2 = x_final[-6:-3]
    t2 = x_final[-3:]
    t2 = np.reshape(t2,(3,1))
    R2 = rodrigues(np.reshape(r2,(r2.size,1)))
    R2 = np.reshape(R2,(3,3))
    M2 = np.hstack((R2,t2))

    return M2, P2

if __name__ == '__main__':
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')    

    corr_pts = np.load('../data/some_corresp.npz')
    pts1 = corr_pts['pts1']
    pts2 = corr_pts['pts2']
    M = max(im1.shape)

#    2.1
    # F = eightpoint(pts1,pts2,M)
#    np.savez('q2_1.npz',F=F,M=M)
#    H.displayEpipolarF(im1,im2,F)

#    2.2
#     pts = np.hstack((pts1,pts2))
#     pts = np.random.permutation(pts)[:7]
#     pts1 = pts[:,0:2]
#     pts2 = pts[:,2:]
#     Farray = sevenpoint(pts1,pts2,M)
# #    np.savez('q2_2.npz',F=Farray[0],pts1=pts1,pts2=pts2,M=M)
#     print(Farray[0])
#     H.displayEpipolarF(im1,im2,Farray[0])
#    
#    3.1/3.2
#    look in findM2.py

#    4.1
    # np.savez('q4_1.npz',F=F,pts1=pts1,pts2=pts2)
    # H.epipolarMatchGUI(im1,im2,F)

#    4.2
#    look in visualize.py
    

    # 5.1
    noisy_pts = np.load('../data/some_corresp_noisy.npz')
    pts1 = corr_pts['pts1']
    pts2 = corr_pts['pts2']
    F,inliers = ransacF(pts1,pts2,M)
    H.displayEpipolarF(im1,im2,F)
#    5.3
#   look in bundleAdjustment.py
