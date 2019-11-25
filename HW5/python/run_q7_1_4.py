import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')

max_iters = 5 # number of epochs
batch_size = 30 # size of batches to train over
learning_rate = .01

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

#MNIST x data is 1 channel images of size 1 x 28 x 28
#MNIST y data is tensors of size N with each element corresponding to
# the digit represented in that image
train_data = torchvision.datasets.EMNIST(root='../data', train=True,
                                      download=True, transform=torchvision.transforms.ToTensor(),
                                      split='balanced')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_data = torchvision.datasets.EMNIST(root='../data', train=False,
                                     download=True, transform=torchvision.transforms.ToTensor(),
                                     split='balanced')
test_loader = DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

c1_out = 2
c2_out = 4
out_size = 47 #this dataset only deals with handwritten digits
hidden_size = 64 

# define Net class that will be our neural network
# net architecture comes from slide 6 of http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture09.pdf
class Net(nn.Module):

    def __init__(self):
        # inheritance
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=c1_out,kernel_size=5,stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2,kernel_size=2))
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(c1_out,c2_out,5,1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2,2))
        
        
        # fully connected layer 1. input of in_size, output of hidden_size
        self.fc1 = nn.Linear(c2_out*4*4,hidden_size)
        
        # fully connected layer2. input of hidden_size, output of out_size
        self.fc2 = nn.Linear(hidden_size,out_size)

    def forward(self,x):

        # 1x28x28 --> c1_outx24x24 (5x5 conv cuts off 2 rows and cols from each end)
        # --> c1_outx12x12 (max pool divides H and W by stride)
        x = self.conv1(x)
#        print("post conv1:",x.size())        

        # c1_outx12x12 --> c2_outx8x8 (5x5 conv cuts off 2 rows and cols from each end)
        # --> c2_outx4x4 (max pool divides H and W by stride)
        x = self.conv2(x)
#        print("post conv2:",x.size())
        
        # reshape x to be N x (c2_outx4x4)
        x = x.view(-1,c2_out*4*4)
#        print("post reshape:", x.size())
        
        # linear transformation from N x last size to N x hidden_size
        x = self.fc1(x)
#        print("post fc1:",x.size())
        
        # linear transformation from N x hidden_size to N x out_size
        x = self.fc2(x)
#        print("post fc2:",x.size())
        
        return x


# define model as our neural network
model = Net()

epochs = list()
accuracies = list()
losses = list()


# define a stochastic gradient descent optimizer with desired learning rate
# and momentum constant of 0.9 (same as problem 5)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
for itr in range(max_iters):
    epochs.append(itr)
    
    total_loss = 0
    avg_acc = 0
    for x_batch, actual in train_loader:
        
        # run model to get a batch_size x out_size tensor. maximum argument
        # in each row corresponds to the class for that image. 
        out = model(x_batch)

        # torch.max returns 2 tensors. [0] is all the max values along the
        # specified axis. [1] is the corresponding indices of the maximum
        # values along the specified axis
        pred = torch.max(out,axis=1)[1]
        
        # count how many arguments of actual and pred match, and divide
        # that by the size of pred to get accuracy
        avg_acc += torch.sum(torch.eq(actual,pred)).item() / pred.size()[0]
        
        # calculate cross entropy loss and add it to total loss. the output of 
        # cross_entropy is a tensor w/ 1 scalar value and a gradient fxn.
        # loss.item() picks off the scalar value from the tensor
        loss = F.cross_entropy(out,actual)
        total_loss += loss.item()

        # performs gradient fxn
        loss.backward()

        # perform one step of SGD and then zero the gradients for the next iteration
        optimizer.step()
        optimizer.zero_grad()

    # get avg loss across all batches
    total_loss /= len(train_loader)
    losses.append(total_loss)
    
    # get avg accuracy across all batches
    avg_acc /= len(train_loader)
    accuracies.append(avg_acc)
    
    print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
    
# run a forward pass on the test data and compute accuracy
test_acc = 0
for x_data, actual in test_loader:
    out = model(x_data)
    pred = torch.max(out,axis=1)[1]
    test_acc += torch.sum(torch.eq(actual,pred)).item() / pred.size()[0]
    
test_acc /= len(test_loader)

print("Test Accuracy:", test_acc)

# plot plots
plt.plot(epochs,losses)
plt.title('Loss over epochs')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.plot(epochs,accuracies)
plt.title('Accuracy over epochs')
plt.xlabel('iteration')
plt.ylabel('accuracy (1 = 100%)')
plt.show()

    


######################## RUN ON IMAGES #####################################
import matplotlib.patches
import os 
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from q4 import *
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
        letter = skimage.transform.resize(letter, (28, 28)).T

        # no need to perform erosion here because the training letters
        # had thickness similar to these letters
#        letter = skimage.morphology.dilation(letter,np.ones((3,3)))
        
        # training images are white on black bg, and these images are black
        # on white bg
        letter= 1-letter

        # flatten and add to the letter_list which will get fed into NN
        letter = letter.flatten()
        letter_list.append(letter)
        
    # compile list into array that is shape (num_letters x img_size). in this case
    # img_size is 784 (28x28)
    letter_list = np.vstack(letter_list)
    letter_list = torch.tensor(letter_list).reshape(-1,1,28,28).float()

    # load the weights
    # run the crops through your neural network and print them out
    import string
    letters = np.array([str(_) for _ in range(10)]+[_ for _ in string.ascii_uppercase[:26]]+["a","b","d","e","f","g","h","n","q","r","t"])
    
    # run a forward pass thru the network to get our predicted letters
    out = model(letter_list)
    
    # convert pred_letters into an array of numbers 0-46 
    pred_letters = torch.max(out,axis=1)[1]
    
    
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
