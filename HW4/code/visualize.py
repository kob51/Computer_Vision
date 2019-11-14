'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import submission as S
import findM2 as find
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png') 

corr_pts = np.load('../data/some_corresp.npz')
corr_pts1 = corr_pts['pts1']
corr_pts2 = corr_pts['pts2']
M = max(im1.shape)

F = S.eightpoint(corr_pts1,corr_pts2,M)



testpts = np.load('../data/templeCoords.npz')

x1 = testpts['x1'].flatten()
y1 = testpts['y1'].flatten()

pts1= np.vstack((x1,y1)).T

pts2 = list()

for i in range(x1.size):
    x2, y2 = S.epipolarCorrespondence(im1,im2,F,x1[i],y1[i])
    pt = np.array([x2,y2])
    pts2.append(pt)
    
pts2 = np.vstack(pts2)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']

M2,C2,P = find.test_M2_solution(pts1, pts2, intrinsics, M)

M1 = np.hstack((np.eye(3),np.zeros((3,1))))
C1 = np.matmul(K1,M1)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(P[:,0],P[:,1],P[:,2])

plt.show()

#np.savez('q4_2.npz', F=F,M1=M1,M 2=M2,C1=C1,C2=C2)