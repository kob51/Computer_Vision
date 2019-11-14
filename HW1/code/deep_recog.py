import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import skimage.io
import scipy

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")
    files = train_data['files']
    labels = train_data['labels']
    
    mypath="../data/"

    pool = multiprocessing.Pool(num_workers)    
    
    
    #check the current configuration of vgg16.classifier to make sure it has
    #the correct number of channels. if there are 7, then we need to 
    #remove the last 3 layers of the vgg network so that the last layer
    #we run is the fc7 layer (second linear layer). if we don't have 4 layers
    #then re-initialize the network and get rid of the last 3 layers.
    
    #try print(vgg16) to see what the network structure looks like
    if len(vgg16.classifier) == 7:
        vgg16.classifier = vgg16.classifier[:-3]
    elif len(vgg16.classifier) != 4:
        vgg16 = torchvision.models.vgg16(pretrained=True).double()
        vgg16.eval()
        vgg16.classifier = vgg16.classifier[:-3] 
    
    args = list()
    
    i=0
    for f in files:
        args.append((i,mypath+f,vgg16))
        i+=1

    print("start multi",time.localtime())
    features = pool.map(get_image_feature,args)
    print("stop multi",time.localtime())
    
    #stack the features to make an array of size (num_imgs x K)
    features = np.vstack(features)

    np.savez("trained_system_deep.npz",features=features,labels=labels)
    
    
    pass

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''
    
    #check the current configuration of vgg16.classifier to make sure it has
    #the correct number of channels. if there are 7, then we need to 
    #remove the last 3 layers of the vgg network so that the last layer
    #we run is the fc7 layer (second linear layer). if we don't have 4 layers
    #then re-initialize the network and get rid of the last 3 layers.
    if len(vgg16.classifier) == 7:
        vgg16.classifier = vgg16.classifier[:-3]
    elif len(vgg16.classifier) != 4:
        vgg16 = torchvision.models.vgg16(pretrained=True).double()
        vgg16.eval()
        vgg16.classifier = vgg16.classifier[:-3]

    conf = np.zeros((8,8))
    
    #load test files and ground truth labels for those files
    test_data = np.load("../data/test_data.npz")
    test_files = test_data['files']
    gt_test_labels = test_data['labels']
    mypath = "../data/"
    
    #initialize array that will store the guessed labels for the test images
    guess_test_labels = np.zeros(gt_test_labels.shape,dtype=int)
    
    trained_system = np.load("trained_system_deep.npz")
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    
    
    #see documentation for build_recognition_system -- same thing done here
    pool = multiprocessing.Pool(num_workers)
    args = list()
    
    i=0
    for f in test_files:
        args.append((i,mypath+f,vgg16))
        i+=1
    test_features = pool.map(get_image_feature,args)
    
    test_features = np.vstack(test_features)
    
#    compare the distance between each test image's feature and the 
#    matrix containing all the training images' features. the smallest value in the
#    return matrix corresponds to the image in the train data that most
#    closely matches the test image. assign the label of that trained image
#    to the test image
    for r in range(test_features.shape[0]):
        guess_test_labels[r] = trained_labels[np.argmin(distance_to_set(test_features[r],trained_features))]
    
    #conf[i][j] represents the number of instances of i that were predicted as j
    for i in range(guess_test_labels.size):
        conf[gt_test_labels[i]][guess_test_labels[i]] += 1

    accuracy = np.trace(conf)/conf.sum()
    
    return conf, accuracy

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''
    
    #have to define this operator as its own variable, for some reason
    #you can't put an argument in those parentheses
    tensorize = torchvision.transforms.ToTensor()
    image_processed = tensorize(image)

    return image_processed

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''
    
    #i don't use the i parameter here. it was more necessary earlier when
    #i was saving temp files with features. 
    i, image_path, vgg16 = args
    image = skimage.io.imread(image_path)
    
    #specify new dimensions
    vgg_img_height = 224
    vgg_img_width = 224
    
    #this fxn normalizes all pixels as well as resizes
    image = skimage.transform.resize(image,(vgg_img_height,vgg_img_width,image.shape[2]))
    
    #adds an empty dimension along axis 0 -- proper formatting for vgg16
    image = preprocess_image(image).unsqueeze(0)  #.detach()
    
    #convert feat back into numpy array before returning
    feat = vgg16(image).detach().numpy()

    return feat

def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    #calculate the euclidian distance between the given feature
    #and each of the features in train_feautures
    feature = np.reshape(feature,(1,-1))
    dist = scipy.spatial.distance.cdist(feature,train_features)#'euclidean'
    
    return dist