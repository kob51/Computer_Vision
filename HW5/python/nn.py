import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    
    lower = -np.sqrt(6/(in_size+out_size))
    upper = -lower
    W = np.random.uniform(lower,upper,(in_size,out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """

    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # y = XW + b
    pre_act = np.matmul(X,W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    
    probs = list()
    
    # perform softmax for each row to get the probability distribution over
    # each class
    for i in range(x.shape[0]):
        # find maximum of x and translate x by -max
        max = np.max(x[i,:])
        temp = x[i,:] - max
   
        #get exponential of each element, calculate sum of all of these exponentials
        e = np.exp(temp)
        s = np.sum(e)
        
        # divide each element in e by the total sum of all expoentials s
        res = e/s
        
        probs.append(res)
    
    res = np.vstack(probs)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):

    # cross-entropy loss formula is in hw5.pdf 
    loss = -np.sum(y*np.log(probs))
    
    num_correct = 0
    for i in range(y.shape[0]):
        # if the maximum probability coincides with the maximum y (y is a 1-hot
        # vector) then we have a correct prediction
        if np.argmax(y[i,:]) == np.argmax(probs[i,:]):
            num_correct += 1
    
    # accuracy is number correct divided by total number of examples
    acc = num_correct/y.shape[0]

    return loss, acc 


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """

    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    d = activation_deriv(post_act)
    
    grad_W = np.matmul(X.T,d*delta)
    grad_b = np.matmul(np.ones((1,delta.shape[0])),d*delta).flatten()
    
    grad_X = np.matmul(d*delta,W.T)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    vals = np.hstack((x,y))
    vals = np.random.permutation(vals)
    
    split = x.shape[1]
    x = vals[:,:split]
    y = vals[:,split:]
    
    for i in range(x.shape[0]//batch_size):
        start = int(i*batch_size)
        end = int((i+1)*batch_size)
        batches.append((x[start:end,:],y[start:end,:]))

    return batches
