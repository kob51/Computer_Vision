import numpy as np
import cv2
import helper as H
import submission as S 
import findM2 as find
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')
M = max(im1.shape)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# load noisy correspondence data (~75% are inliers)
noisy = np.load('../data/some_corresp_noisy.npz')
pts1 = noisy['pts1']
pts2 = noisy['pts2']

# perform RANSAC to get the best F and inliers
F, inliers = S.ransacF(pts1,pts2,M)

# filter out outliers from pts1 and pts2
pts1 = pts1[inliers,:]
pts2 = pts2[inliers,:]

# get best M2, and corresponding C2 and P from the points
M2_init,C2_init,P_init = find.test_M2_solution(pts1, pts2, intrinsics, M)

# C1 is just the identity in rotation and no translation
M1 = np.hstack((np.eye(3),np.zeros((3,1))))
C1 = np.matmul(K1,M1)

# compute a new M2 and set of points P from bundle adjustment. get 
# the corresponding camera matrix C2
M2, P = S.bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_init)
C2 = np.matmul(K2,M2)

# triangulate the points using this new C2
P_ba, error = S.triangulate(C1, pts1, C2, pts2)
print("Error after bundle adjustment:",error)

fig = plt.figure()
ax = Axes3D(fig)

# blue --> initial P, after finding best M2 from test_M2_solution
# red --> P output from triangulating the inliers using the new C2 from bundle adjustment

# yellow --> P output from bundle adjustment (?)
ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c='b')
ax.scatter(P_ba[:, 0], P_ba[:, 1], P_ba[:, 2], c='r')
# ax.scatter(P[:,0],P[:,1],P[:,2],c='y')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

