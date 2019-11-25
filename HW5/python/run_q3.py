import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50 # number of epochs
batch_size = 30 # size of batches to train over
learning_rate = .01

hidden_size = 64 # output size of hidden layer (layer1)
out_size = 36 # output size of network (26 letters + 10 digits)

# initialize batches of training data
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,out_size,params,'output')

# visualize initialized weights here
# W1 = params['Wlayer1']
# fig = plt.figure()
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8))# creates 2x2 grid of axes

# for i in range(W1.shape[1]):
#     test = W1[:,i]
#     test = np.reshape(test,(32,32))
#     grid[i].imshow(test)
#     plt.axis('off')

# plt.show()

valid_accuracies = list()
valid_losses = list()
train_accuracies = list()
train_losses = list()
epochs = list()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    epochs.append(itr)
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        
        # forward
        h1 = forward(xb,params,'layer1',sigmoid)
        probs = forward(h1,params,'output',softmax)
        
        # loss 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        avg_acc += acc
        
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        
        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    # compute average accuracy for this epoch
    avg_acc /= batch_num
    total_acc = avg_acc

    # compute total loss for this epoch
    total_loss /= batch_num
    train_losses.append(total_loss)
    train_accuracies.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%

    # in each epoch, do a forward pass with the validation data and get the
    # accuracy and loss
    valid_h1 = forward(valid_x,params,'layer1')
    valid_probs = forward(valid_h1,params,'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y,valid_probs)
    valid_losses.append(valid_loss/batch_num)
    valid_accuracies.append(valid_acc)
    
print('Validation accuracy: ',valid_acc)

# graph accuracies for training and validation
fig,ax = plt.subplots()
plt.plot(epochs,train_accuracies,label='training accuracy')
plt.plot(epochs,valid_accuracies,label='validation accuracy')
ax.legend()
plt.title('Accuracy over epochs')
plt.xlabel('iteration')
plt.ylabel('accuracy (1 = 100%)')
plt.show()

# graph losses for training and validation
fig1,ax1 = plt.subplots()
plt.plot(epochs,train_losses,label='training loss')
plt.plot(epochs,valid_losses,label='validation loss')
ax1.legend()
plt.title('Loss over epochs')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3

# visualize weights here
W1 = params['Wlayer1']
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8))# creates 2x2 grid of axes

for i in range(W1.shape[1]):
    test = W1[:,i]
    test = np.reshape(test,(32,32))
    grid[i].imshow(test)
    plt.axis('off')

plt.show()
    

# Q3.1.4
confusion_matrix = np.zeros((test_y.shape[1],test_y.shape[1]))

#run a forward pass using test data
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)

# compute confusion matrix here
test_loss, test_acc = compute_loss_and_acc(test_y,test_probs)
print("Test Accuracy: ", test_acc)

for i in range(test_y.shape[0]):
    actual = np.argmax(test_y[i,:])
    pred = np.argmax(test_probs[i,:])

    confusion_matrix[actual,pred] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()