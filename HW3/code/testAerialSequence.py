import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

frames = np.load('../data/aerialseq.npy')
#import datetime
#print(datetime.datetime.now())

mask_list = list()
for i in range(frames.shape[2]-1):
    
    mask = SubtractDominantMotion(frames[:,:,i],frames[:,:,i+1])
    
    if i == 30 or i == 60 or i == 90 or i ==120:
        im = np.dstack((frames[:,:,i],frames[:,:,i],frames[:,:,i]))
        mask_list.append(mask)
#        im[mask,2] = 1
#        print(i)
#        plt.imshow(im)
#        plt.show()
        
#print(datetime.datetime.now())       
mask_list = np.dstack(mask_list)
np.save('aerialseqmasks.npy',mask_list)
