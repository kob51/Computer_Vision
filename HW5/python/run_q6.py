import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

# PCA algorithm from 
# https://www.coursera.org/lecture/machine-learning/principal-component-analysis-algorithm-ZYIPa

dim = 32

# do PCA
# subtract the mean from each element of train_x (mean normalization)
train_mean = np.sum(train_x, axis=0)/train_x.shape[0]
train_x -= train_mean
print("train_x:",train_x.shape)

# sigma is the covariance matrix
sigma = np.matmul(train_x.T,train_x)/train_x.shape[0]

# compute SVD on sigma
u,s,vh = np.linalg.svd(sigma)

# the projection matrix is the first "dim" columns of U
projection = u[:,:dim]

print("Projection matrix size:", projection.shape) # 1024 x 32
print("Projection matrix rank:", np.linalg.matrix_rank(projection)) #32

#the projection matrix takes in an N x 1024 matrix and returns and N x 32 matrix,
#effectively reducing the dimensions of the data from 1024 to 32. our projection
#matrix can be thought of as the "trained" weights. we can take an image of 1024 dimensions,
#project it to 32 dimensions by multiplying (image)x(projection), and reconstruct
#the original image by multiplying that result by projection.T, which takes in
#32 dimensions and casts it back out to 1024

# rebuild a low-rank version (cast to 32 dimensions)
lrank = np.matmul(train_x,projection)
# print("lrank:",lrank.shape)

# rebuild it (cast back to 1024 dimensions)
recon = np.matmul(lrank, projection.T)
# print("recon:",recon.shape)

# build valid dataset
# perform mean normalization on valid
valid_mean = np.sum(valid_x, axis=0)/valid_x.shape[0]
valid_x -= valid_mean

# reconstruct the valid images by multiplying by projection (cast to 32) and 
# projection.T (cast back to 1024)
recon_valid = np.matmul(np.matmul(valid_x, projection), projection.T)

# undo mean normalization
recon_valid += valid_mean
valid_x += valid_mean




# print 2 instances of K, S, O, B, 2
test_list = [1000,1001,1800,1801,1400,1401,100,101,2800,2801]
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 4))# creates 2x2 grid of axes
i=0
for letter in test_list:
    actual = valid_x[letter].reshape(32,32).T
    grid[i].imshow(actual)
    i +=1
    pred = recon_valid[letter].reshape(32,32).T
    grid[i].imshow(pred) 
    i+=1
plt.show()

# calculate avg Peak Signal-to-noise Ratio (PSNR) over all the images
avg_psnr = 0
for i in range(valid_x.shape[0]):
    actual = valid_x[i].reshape(32,32).T
    pred = recon_valid[i].reshape(32,32).T
    avg_psnr += psnr(actual,pred)

avg_psnr /= valid_x.shape[0]
print("Average PSNR:", avg_psnr)
