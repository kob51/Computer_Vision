import numpy as np
import submission as S
import helper as H
import cv2
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics, M):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :param M: a scalar parameter computed as max (imwidth, imheight)
    :return: M2, the extrinsics of camera 2
            C2, the 3x4 camera matrix
            P, 3D points after triangulation (Nx3)
    '''
    # get fundamental matrix
    F = S.eightpoint(pts1,pts2,M)
    
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    
    # get essential matrix
    E = S.essentialMatrix(F,K1,K2)
    
    # create the camera matrix for camera 1
    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    C1 = np.matmul(K1,M1)
    
    # get the possible extrinsic matrices for camera 2 using E
    M_list = H.camera2(E)
    
    min_err = -1
    # test the 4 options for M2, and keep the one that produces the minimum error
    for i in range(4):
        
        test_M2 = M_list[:,:,i]
        test_C2 = np.matmul(K2,test_M2)
        test_P,test_err = S.triangulate(C1, pts1, test_C2, pts2)
        
        
        # if the minimum z position is positive, that means that this solution
        # for M2 gives a 3d point that is in front of both cameras, so we've found
        # our solution
        if np.min(test_P[:,2]) > 0 and (test_err < min_err or min_err == -1):
            M2 = test_M2
            C2 = test_C2
            P = test_P
            min_err = test_err
            
    print("Error before bundle adjustment:",min_err)        
    return M2, C2, P


if __name__ == '__main__':
    im1 = cv2.imread('../data/im1.png')
    M = max(im1.shape)
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    intrinsics = np.load('../data/intrinsics.npz')

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics, M)
    np.savez('q3_3', M2=M2, C2=C2, P=P)
