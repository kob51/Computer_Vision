import numpy as np
import scipy.ndimage
import os
import torchvision
import skimage
import skimage.transform

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.
    
    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)
    
    [output]
    * feat: numpy.ndarray of shape (K)
    '''    
    
    #reshape image to required height for vgg net
    vgg_img_height = 224
    vgg_img_width = 224
    x = skimage.transform.resize(x,(vgg_img_height,vgg_img_width,x.shape[2]))
    
    #normalize image using specified mean and std dev
    mean = [0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    for i in range(x.shape[2]):
        x[:,:,i] = (x[:,:,i] - mean[i])/std[i]
        
    #loop thru each layer of the network and perform the specified calculations
    # on the image
    for w in vgg16_weights[:-2]:
        if w[0] == "conv2d":
            x = multichannel_conv2d(x,w[1],w[2])
        elif w[0] == "relu":
            x = relu(x)
        elif w[0] == "maxpool2d":
            x = max_pool2d(x,w[1])
        elif w[0] == "linear":
            x = linear(x.flatten(),w[1],w[2])

    return x


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''
    
    output = list()
    #loop thru all of the weights (output_dim). each weight has input_dim (think 3 
    #for initial image) filters, one for each channel. in each weight, run each 
    #filter on the corresponding channel, and then add the input_dim responses
    #together so you get a HxW result. Stack each result of this outer loop
    #to get a HxWxoutput_dim array. See formula for multichannel convolution
    #for clarification
    for w in range(weight.shape[0]):
        channels = list()
        for i in range(weight.shape[1]):
            y = np.flip(weight[w,i,:,:],(0,1)) #flip kernel for convolution
            temp = scipy.ndimage.convolve(x[:,:,i],y,mode='constant',cval=0)
            channels.append(temp)
        channel_array = np.asarray(channels)
        result = np.sum(channel_array,axis=0)
        output.append(result)
    
    feat = np.dstack(output) + bias

    return feat

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    
    #ReLU takes keeps all elements except negative ones. Negative values
    #are mapped to zero
    #f(x) = max(0,x)
    y = np.maximum(x,0)

    return y

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''
    
    #divide the image up into squares of (size)x(size) and save the max element in
    #each square
    channels = list()
    for i in range(x.shape[2]):
        
        rows = list()
        for r in range(int(x.shape[0]/size)):
            
            temp_col = list()
            for c in range(int(x.shape[1]/size)):      
                temp = x[int(r*size):int((r+1)*size),
                               int(c*size):int((c+1)*size),i]
                temp_col.append(np.max(temp))
            
            temp_row = np.hstack(temp_col)
            
            rows.append(temp_row)
        
        temp_channel = np.vstack(rows)
        channels.append(temp_channel)
    
    y = np.dstack(channels)
    
    return y;

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    
    #convert from a vector in input_dim space to a vector in output_dim space
    y = np.matmul(W,x) + b
    
    return y

