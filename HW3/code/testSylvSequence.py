import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis as LKB
import LucasKanade as LK

frames = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')

#frames = frames[:,:,0:401]

# TRUE --> program will create/save a flipbook-esque video
# following the object
# FALSE --> program will output the last frame of the video
# with overlaid rectangles showing tracking with orig LK and basis-corrected LK
save_vid = False

#fig, ax = plt.subplots(1)

x1 = 101
y1 = 61
x2 = 155
y2 = 107

rect_list = list()

# _b denotes LKBasis method
corners_b = np.array([[x1,y1],[x2,y2]],'float')

corners = corners_b.copy()

rect_list.append(corners.flatten())

vid = []

p = np.zeros(2)

for i in range(frames.shape[2]-1):

    p_b = LKB.LucasKanadeBasis(frames[:,:,i], frames[:,:,i+1], corners_b.flatten(),bases)
    
    if not save_vid:
        p = LK.LucasKanade(frames[:,:,i],frames[:,:,i+1],corners.flatten())
        corners[:,0] += p[0]
        corners[:,1] += p[1]  
        x1 = corners[0,0]
        y1 = corners[0,1]
        x2 = corners[1,0]
        y2 = corners[1,1]
        x_len = x2 - x1 + 1
        y_len = y2 - y1 + 1
        
#    print("frame",i)
    
    #add x and y values of p to the corner coordinates
    corners_b[:,0] += p_b[0]
    corners_b[:,1] += p_b[1]
    
    x1_b = corners_b[0,0]
    y1_b = corners_b[0,1]
    x2_b = corners_b[1,0]
    y2_b = corners_b[1,1]
    
    x_len_b = x2_b - x1_b + 1
    y_len_b = y2_b - y1_b + 1
    
    # add the corners to the list we're going to save
    rect_list.append(corners_b.flatten())

    #create a rectangular patch to indicate the location of the car
    rect_b = patches.Rectangle((x1_b,y1_b),x_len_b,y_len_b,linewidth=1,edgecolor='g',facecolor='none')
    
    if save_vid:
        vid.append([ax.imshow(frames[:,:,i], cmap = 'gray', animated = True)])
        vid.append([ax.add_patch(rect_b)])

     
rect_list = np.vstack(rect_list)
np.save('sylvseqrects.npy',rect_list)


if save_vid:
    print("saving video...")
    im_ani = animation.ArtistAnimation(fig,vid,interval=50)
    im_ani.save('sylvseq.mp4')
#else:
#    rect = patches.Rectangle((x1,y1),x_len,y_len,linewidth=1,edgecolor='r',facecolor='none')
#    ax.imshow(frames[:,:,-1],cmap='gray')
#    ax.add_patch(rect)
#    ax.add_patch(rect_b)
#    plt.show()