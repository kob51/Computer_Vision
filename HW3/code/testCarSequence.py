import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK

frames = np.load('../data/carseq.npy')

#frames = frames[:,:,0:401]

# TRUE --> program will create/save a flipbook-esque video
# with rectangle following the object
# FALSE --> program will output the last frame of the video
# with overlaid rectangle showing tracking
save_vid = False

#fig, ax = plt.subplots(1)

x1 = 55
y1 = 116
x2 = 145
y2 = 151

rect_list = list()

corners = np.array([[x1,y1],[x2,y2]],'float')

rect_list.append(corners.flatten())

p = np.zeros(2)

vid = []
for i in range(frames.shape[2]-1):
    
    p = LK.LucasKanade(frames[:,:,i], frames[:,:,i+1], corners.flatten(),p)

#    print("frame",i)
    
    #add x and y values of p to the corner coordinates
    corners[:,0] += p[0]
    corners[:,1] += p[1]
    
    # add the corners to the list we're going to save
    rect_list.append(corners.flatten())
    
    x1 = corners[0,0]
    y1 = corners[0,1]
    x2 = corners[1,0]
    y2 = corners[1,1]
    
    x_len = x2 - x1 + 1
    y_len = y2 - y1 + 1

    #create a rectangular patch to indicate the location of the car
    rect = patches.Rectangle((x1,y1),x_len,y_len,linewidth=1,edgecolor='r',facecolor='none')
    if save_vid:
        vid.append([ax.imshow(frames[:,:,i], cmap = 'gray', animated = True)])
        vid.append([ax.add_patch(rect)])
    
    
rect_list = np.vstack(rect_list)
np.save('carseqrects.npy',rect_list)


if save_vid:
    print("saving video...")
    im_ani = animation.ArtistAnimation(fig,vid,interval=50)
    im_ani.save('carseq.mp4')  
#else:
#    ax.imshow(frames[:,:,-1],cmap='gray')
#    ax.add_patch(rect)
#    plt.show()