import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
out_size = train_x.shape[1]
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(out_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,out_size,params,'output')

epochs=list()
losses=list()

# should look like your previous training loops
for itr in range(max_iters):
    epochs.append(itr)
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        h3 = forward(h2,params,'layer3',relu)
        out = forward(h3,params,'output',sigmoid)
        
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss = np.sum(np.square(out-xb))

        total_loss += loss
        
        # backward
        delta1 = 2 * (out-xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        delta5 = backwards(delta4,params,'layer1',relu_deriv)
        
        # apply gradient

        # calculate momentum for each weight
        params['m_Wlayer1'] = .9*params['m_Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['m_Wlayer2'] = .9*params['m_Wlayer2'] - learning_rate * params['grad_Wlayer2']
        params['m_Wlayer3'] = .9*params['m_Wlayer3'] - learning_rate * params['grad_Wlayer3']
        params['m_Woutput'] = .9*params['m_Woutput'] - learning_rate * params['grad_Woutput']

        # calculate momentum for each bias
        params['m_blayer1'] = .9*params['m_blayer1'] - learning_rate * params['grad_blayer1']
        params['m_blayer2'] = .9*params['m_blayer2'] - learning_rate * params['grad_blayer2']
        params['m_blayer3'] = .9*params['m_blayer3'] - learning_rate * params['grad_blayer3']
        params['m_boutput'] = .9*params['m_boutput'] - learning_rate * params['grad_boutput']

        # update each weight with corresponding momentum
        params['Wlayer1'] += params['m_Wlayer1']
        params['Wlayer2'] += params['m_Wlayer2']
        params['Wlayer3'] += params['m_Wlayer3']
        params['Woutput'] += params['m_Woutput']

        # update each bias with corresponding momentum
        params['blayer1'] += params['m_blayer1']
        params['blayer2'] += params['m_blayer2']
        params['blayer3'] += params['m_blayer3']
        params['boutput'] += params['m_boutput']        

    # average loss over number of batches
    total_loss /= batch_num
    losses.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# plot loss vs epochs
plt.plot(epochs,losses)
plt.title('Loss over epochs')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# Q5.3.1

# forward pass of validation data
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output',sigmoid)

# print 2 instances of K, S, O, B, 2 from validation (pre NN) and post NN
test_list = [1000,1001,1800,1801,1400,1401,100,101,2800,2801]
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 4))# creates 2x2 grid of axes
i=0
for letter in test_list:
    actual = valid_x[letter].reshape(32,32).T
    grid[i].imshow(actual)
    i +=1
    pred = out[letter].reshape(32,32).T
    grid[i].imshow(pred) 
    i+=1
plt.show()


# Q5.3.2
# calculate avg Peak Signal-to-noise Ratio (PSNR) over all the images
avg_psnr = 0
from skimage.measure import compare_psnr as psnr
for i in range(valid_x.shape[0]):
    actual = valid_x[i].reshape(32,32).T
    pred = out[i].reshape(32,32).T
    avg_psnr += psnr(actual,pred)

avg_psnr /= valid_x.shape[0]
print("Average PSNR:", avg_psnr)
