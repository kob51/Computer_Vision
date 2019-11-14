import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import skimage.io

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    
    mypath="../data/"
    
    files = train_data['files']
    labels = train_data['labels']
    layer_num = 3

    pool = multiprocessing.Pool(num_workers)    
    
    args = list()
    
    #initially i tried doing something similar to what was done in 
    #visual_word.compute dictionary with map(). however, found on stack overflow
    #that map() only works well for multiple arguments if those arguments are
    #expected as a list by the function you're calling.
    
    #to work around this i wrote a wrapper that unpacks a list of 
    #arguments and feeds them to get_image_feature so that I can call
    #this function using all my processors and .map(). it still took 22 minutes...
    for f in files:
        args.append((mypath+f,dictionary,layer_num,dictionary.shape[0]))
        
    features = pool.map(img_ftr_map_wrapper,args)

    #stack the features to make an array of size (num_imgs x M)
    features = np.vstack(features)

    np.savez("trained_system.npz",dictionary=dictionary,features=features,
             labels=labels,layer_num=layer_num)
    pass

def img_ftr_map_wrapper(args):
    file_path, dictionary, layer_num, K = args
    return get_image_feature(file_path, dictionary, layer_num, K)

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''
    
    conf = np.zeros((8,8))
    
    #load test files and ground truth labels for those files
    test_data = np.load("../data/test_data.npz")
    test_files = test_data['files']
    gt_test_labels = test_data['labels']
    
    mypath = "../data/"
    
    #initialize array that will store the guessed labels for the test images
    guess_test_labels = np.zeros(gt_test_labels.shape,dtype=int)
    
    trained_system = np.load("trained_system.npz")
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    layer_num = np.asscalar(trained_system['layer_num'])
    
    
    #see documentation for build_recognition_system -- same thing done here
    pool = multiprocessing.Pool(num_workers)
    args = list()
    
    for f in test_files:
        args.append((mypath+f,dictionary,layer_num,dictionary.shape[0]))
    
    test_features = pool.map(img_ftr_map_wrapper,args)
    test_features = np.vstack(test_features)
    
#    compare the similarity between each test image's histogram and the 
#    histogram containing all the training images. the highest value in the
#    return matrix corresponds to the image in the train data that most
#    closely matches the test image. assign the label of that trained image
#    to the test image
    for r in range(test_features.shape[0]):
        guess_test_labels[r] = trained_labels[np.argmax(distance_to_set(test_features[r],trained_features))]
    
    #conf[i][j] represents the number of instances of i that were predicted as j
    for i in range(guess_test_labels.size):
        conf[gt_test_labels[i]][guess_test_labels[i]] += 1

    accuracy = np.trace(conf)/conf.sum()
    return conf, accuracy

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
     
    return feature

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    #COMPUTE HISTOGRAM INTERSECTION SIMILARITY
    #H.I.S. is defiend as the sum of the minimum value in each corresponding bin

    #stack word hist up N times so we get an (N, K) array
    word_hist = np.tile(word_hist,(histograms.shape[0],1))
    
    #element-wise minimum of the two arrays
    minima = np.minimum(word_hist,histograms)
    
    #add up the rows of the column to get the vector H.I.S. between wordmap and each of
    #N histograms
    sim = np.sum(minima, axis = 1)
    
    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = dict_size
    
    #L1 norm is just sum of abs val of elements (Manhattan distance)
    #hist is L1 normalized because each element is divided by L1 norm
    #**the sum of an L1 normalized vector is 1.
    
    #the bins array is range(K+1) because the fxn uses the gaps btwn the elements
    #as bins. for example if bins=[1,2,3,4], the bins are [1,2),[2,3),[3,4].
    #we want K bins so we use range(K+1). if we just used bins=K, the fxn
    #automatically assigns values to the bins. we want one bin for each K cluster
    
    raw_hist,bins = np.histogram(wordmap,range(K+1))
    hist = raw_hist/np.linalg.norm(raw_hist,1)

    return hist

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    #total number of layers
    L = layer_num 
    
    #go in reverse so that we start at the finest channel (channel with 
    #densest grid). in this case, that's channel 2 because that channel
    #gets a 4x4 grid (2**l x 2**l)    
    test = list()
    channels = list(range(L))
    channels.reverse()
    
    #assigns weights to each channel's histograms -- see formula in hw1.pdf
    for l in channels:
        if l < 2:
            weight = 2**(-L)
        else:
            weight = 2**(l-L-1)       
        
        #create grid of 2**l x 2**l cells from wordmap 
        for r in range(2**l):
            for c in range(2**l):           
                temp = wordmap[int(r*wordmap.shape[0]/(2**l)):int((r+1)*wordmap.shape[0]/(2**l)),
                               int(c*wordmap.shape[1]/(2**l)):int((c+1)*wordmap.shape[1]/(2**l))]
                test.append(weight*get_feature_from_wordmap(temp,dict_size))
                
    hist_all = np.hstack(test)
    hist_all = hist_all/np.linalg.norm(hist_all,1)

    return hist_all