#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:36:25 2019

@author: kevinobrien
"""

import cv2
import BRIEF
import matplotlib.pyplot as plt
import numpy as np


im = cv2.imread('../data/model_chickenbroth.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

rows = im.shape[0]
cols = im.shape[1]

matches = list()
angles = list()

locs, desc = BRIEF.briefLite(im)

# rotate the image in increments of 10 degrees and count the number of 
# keypoint matches between the sample image and the rotated image
for i in range(37):
    rot_angle = 10*i
    rot_mat = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    im_rot = cv2.warpAffine(im,rot_mat,(im.shape[1],im.shape[0]))
    
    print("rotated %d degrees" % (rot_angle))

    locs_rot, desc_rot = BRIEF.briefLite(im_rot)
    temp_matches = BRIEF.briefMatch(desc, desc_rot)
    temp_matches = temp_matches.shape[0]
    matches.append(temp_matches)
    angles.append(10*i)
   


# plot the histogram showing number of matches as a function of rotation angle
#index = np.arange(len(angles))
#plt.bar(index, matches,width=1)
#plt.xlabel('Angle of Rotation (deg)', fontsize=10)
#plt.ylabel('Number of Matches', fontsize=10)
#plt.xticks(index, angles, fontsize=8, rotation=90)
#plt.title('Number of matches for each angle')
#plt.show()