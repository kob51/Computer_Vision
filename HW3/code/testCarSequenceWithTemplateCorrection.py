import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK

#template correction logic comes from this paper:
#https://www.ri.cmu.edu/pub_files/pub4/matthews_iain_2003_2/matthews_iain_2003_2.pdf

frames = np.load('../data/carseq.npy')

#frames = frames[:,:,0:401]

# TRUE --> program will create/save a flipbook-esque video
# following the object
# FALSE --> program will output the last frame of the video
# with overlaid rectangles showing tracking with orig LK and drift-corrected LK
save_vid = False

#frames = np.dstack((frames[:,:,50:55],frames[:,:,100:130]))

#fig, ax = plt.subplots(1)

x1 = 55
y1 = 116
x2 = 145
y2 = 151

rect_list = list()

initial_corners = np.array([[x1,y1],[x2,y2]],'float')
corners = initial_corners.copy()

rect_list.append(corners.flatten())

# corners without template correction ("no drift")
corners_nd = initial_corners.copy()

vid = []

total_p = np.zeros(2)
pstar = np.zeros(2)
p = np.zeros(2)

for i in range(frames.shape[2]-1):
    
    # get p between 2 consecutive frames
    p = LK.LucasKanade(frames[:,:,i], frames[:,:,i+1], corners.flatten(),p)
    
    # add this p to my total_p that keeps track of the total movement from 
    # frame zero to current frame
    total_p = np.add(total_p,p)
    
    # using total_p as an initial guess, find the p btwn frame 0 and current frame
    pstar = LK.LucasKanade(frames[:,:,0], frames[:,:,i+1], initial_corners.flatten(),total_p)
    
    #calculate error btwn pstar(predicted) and total_p(actual)
    drift = pstar - total_p
#    print("frame",i)
    tol = 5
    
    # if the drift is negligible (magnitude of 5 corresponds to error of
    # ~3.5 pixels in each direction), then we update both the total_p and the
    # rectangle by adding the calculated drift. if the drift is too high,
    # there must be a problem, so we act conservatively by not updating
    # the rectangle or total_p by the drift
    if np.linalg.norm(drift) <= tol:    
        corners[:,0] += p[0] + drift[0]
        corners[:,1] += p[1] + drift[1]
        total_p = np.add(total_p,drift)
    else:
        corners[:,0] += p[0]
        corners[:,1] += p[1]
        
    # always update corners_nd with the newly computed value of p,
    # just like we did in the naive LK implementation
    corners_nd[:,0] += p[0]
    corners_nd[:,1] += p[1]

    # add the current value of corners to our rect_list that we'll save
    rect_list.append(corners.flatten())
    
    x1 = corners[0,0]
    y1 = corners[0,1]
    x2 = corners[1,0]
    y2 = corners[1,1]
    
    x_len = x2 - x1 + 1
    y_len = y2 - y1 + 1
    
    x1_nd = corners_nd[0,0]
    y1_nd = corners_nd[0,1]
    x2_nd = corners_nd[1,0]
    y2_nd = corners_nd[1,1]
    
    x_len_nd = x2_nd - x1_nd + 1
    y_len_nd = y2_nd - y1_nd + 1
    
    # create flipbook animation
    rect = patches.Rectangle((x1,y1),x_len,y_len,linewidth=1,edgecolor='g',facecolor='none')
    if save_vid:
        vid.append([ax.imshow(frames[:,:,i], cmap = 'gray', animated = True)])
        vid.append([ax.add_patch(rect)])


rect_list = np.vstack(rect_list)
np.save('carseqrects-wcrt.npy',rect_list)


if save_vid:
    print("saving video...")
    im_ani = animation.ArtistAnimation(fig,vid,interval=50)
    im_ani.save('carseqwcrt.mp4')
#else:
#    ax.imshow(frames[:,:,-1],cmap='gray')
#    ax.add_patch(rect)    
#    rect_nd = patches.Rectangle((x1_nd,y1_nd),x_len_nd,y_len_nd,linewidth=1,edgecolor='r',facecolor='none')
#    ax.add_patch(rect_nd)
#    plt.show()