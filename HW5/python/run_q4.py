import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import scipy.io

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

train_data = scipy.io.loadmat('../data/nist36_train.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']


# x and y padding for test images
x_pad = 15
y_pad = 15


i = 0
for img in sorted(os.listdir('../images')):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    rows,cols,_ = im1.shape
        
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

#   first, sort all bounding boxes by height
    bboxes = sorted(bboxes,key=lambda x:x[:][0])

    # loop thru all bounding boxes, and build a list of box coordinates along
    # with the current row that we're at. increment the row anytime the y location
    # of the centroid has increased by more than the centroid height.
    last_centroid = -1
    curr_row = 0
    row_list = list()
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        height = (y2-y1)/2
        centroid = (y1+y2)/2
        if centroid - last_centroid > height or last_centroid == -1:
            curr_row += 1
        last_centroid = centroid
        row_list.append((curr_row,y1,x1,y2,x2))
        
    # now sort the bounding boxes first by row and second by x coordinate.
    # this should put the letters in the order that humans read them
    bboxes = sorted(row_list,key=lambda x: (x[:][0],x[:][2]))


    letter_list = list()
    for bbox in bboxes:
        _, y1, x1, y2, x2 = bbox
        
        # crop out the letter from the image
        letter = bw[y1:y2+1,x1:x2+1]
        
        # pad with ones so that the cropped image is closer to the format of 
        # the training images
        letter = np.pad(letter, ((x_pad, x_pad), (y_pad, y_pad)), 'constant', 
                        constant_values=(1, 1))

        # resize to 32x32 and transpose (to fit the format of training imgs)
        letter = skimage.transform.resize(letter, (32, 32)).T

        # perform erosion on the image (erodes white parts) to effectively
        # dilate the text. the training images have thick letters while some
        # of these test images have thinner letters
        letter = skimage.morphology.erosion(letter,np.ones((3,3)))

        # flatten and add to the letter_list which will get fed into NN
        letter = letter.flatten()
        letter_list.append(letter)
        
    # compile list into array that is shape (num_letters x img_size). in this case
    # img_size is 1024 (32x32)
    letter_list = np.vstack(letter_list)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    # run a forward pass thru the network to get our predicted letters
    h1 = forward(letter_list,params,name='layer1')
    pred_letters = forward(h1,params,'output',softmax)
    
    # convert pred_letters into an array of numbers 0-35, corresponding to the
    # predicted character. the max argument in each column. argmax picks the 
    # highest probability from softmax.
    pred_letters = np.argmax(pred_letters,axis=1)
    
    if img == "03_haiku.jpg":
        gt = "HAIKUS ARE EASY\nBUT SOMETIMES THEY DONT MAKE SENSE\nREFRIGERATOR"
    if img == "04_deep.jpg":
        gt = "DEEP LEARNING\nDEEPER LEARNING\nDEEPEST LEARNING"
    if img == "01_list.jpg":
        gt = "TO DO LIST\n1 MAKE A TO DO LIST\n2 CHECK OFF THE FIRST\nTHING ON TO DO LIST\n3 REALIZE YOU HAVE ALREADY\nCOMPLETED 2 THINGS\n4 REWARD YOURSELF WITH\nA NAP"
    if img == "02_letters.jpg":
        gt = "ABCDEFG\nHIJKLMN\nOPQRSTU\nVWXYZ\n1234567890"

    #build string of predicted letters and compare to ground truth
    string =''
    j=0
    for i in range(len(gt)):
        if gt[i] == '\n':
            string += '\n'
        elif gt[i] == ' ':
            string += ' '
        else:
            string += letters[pred_letters[j]]
            j+=1
    print("NN OUTPUT:")
    print(string,'\n')

        
    print("EXPECTED OUTPUT:")
    print(gt,'\n')
    
    acc = 0
    for i in range(len(string)):
        if string[i] == '\n' or string[i] == ' ':
            continue
        elif string[i] == gt[i]:
            acc +=1
    acc/=pred_letters.shape[0]
    print(img,"accuracy:", acc,'\n')
