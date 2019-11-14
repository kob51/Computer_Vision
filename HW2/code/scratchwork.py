#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:23:56 2019

@author: kevinobrien
"""
import datetime
path = "../data/chickenbroth_05.jpg"
import keypointDetect as kd
import cv2
import numpy as np
image = cv2.imread(path)
import planarH
import ar


#gp = kd.createGaussianPyramid(image)
##kd.displayPyramid(gp)
#dog = kd.createDoGPyramid(gp)[0]

#kd.displayPyramid(dog)
#test = dog[:,:,0]
#testxy = cv2.normalize(test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#testxy = cv2.Sobel(testxy,ddepth=-1, dx=0,dy=1)
#testxy = cv2.Sobel(testxy,ddepth=-1,dx=1,dy=0)
#cv2.imshow('Dxy',testxy)
#
#testyx = cv2.normalize(test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#testyx = cv2.Sobel(testyx,ddepth=-1, dx=1,dy=0)
#testyx = cv2.Sobel(testyx,ddepth=-1,dx=0,dy=1)
#cv2.imshow('Dyx',testyx)
#
#test_simul = cv2.normalize(test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#test_simul = cv2.Sobel(test_simul,ddepth=-1, dx=1,dy=1)
#cv2.imshow('simul',test_simul)




locsDoG = kd.DoGdetector(image)[0]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
i = 0
for point in locsDoG:
    cv2.circle(image,tuple(np.flip(point[:-1])),1,(0,255,0))
    i+=1
cv2.imshow("feature points",image)
#
cv2.waitKey(0)
cv2.destroyAllWindows()

#[:,:8]

#homo = planarH.computeH(test2,test2)
#image = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
##H = H/H[2,2]
#testwarp = cv2.warpPerspective(image,H,(image.shape[1],image.shape[0]))
#actualwarp = cv2.warpPerspective(image,expected_H,(image.shape[1],image.shape[0]))
#cv2.imshow("warped soup",testwarp)
#cv2.imshow("normal soup",image)
#cv2.imshow("expected soup",actualwarp)
#im2 = cv2.imread('../data/incline_R.png')
#im2_warp = cv2.warpPerspective(im2,np.linalg.inv(M),(im2.shape[1],im2.shape[0]))
cv2.imshow("pano?",pano_im)
#cv2.imshow("im L",im1)
##cv2.imshow("im R",im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

