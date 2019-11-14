import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import skimage.io

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''    

    # make sure image has 3 channels, is a float, and is btwn 0 and 1
    if len(image.shape) < 3:
        image = np.stack((image,image,image))
        print("image stacked")

    if image.dtype != float:
        image = image.astype('float')

    if image.max() > 1:
        image = image/255

    # convert to LAB colorspace --> more effective to quantify color diffs
    image = skimage.color.rgb2lab(image)

    #initialize placeholder for all filter responses
    responses=list()
    
    for s in (1,2,4,8,(8)*2**(1/2)):
        for filt_type in ("G","L_G","G_X","G_Y"):  
            
            #Gaussian filter on all 3 channels
            if filt_type == "G":
                test_img = scipy.ndimage.gaussian_filter(image,sigma=(s,s,0))
                
            #Laplace of Gaussian filter on all 3 channels
            elif filt_type == "L_G":
                
                test = list()
                for r in range(3):
                    test_img = scipy.ndimage.gaussian_laplace(image[:,:,r],sigma=s)
                    test.append(test_img)
                test_img = np.dstack(test)
                
            #derivative of Gaussian filter in the x direction
            elif filt_type == "G_X":
                test_img = scipy.ndimage.gaussian_filter(image,sigma=(s,s,0),order=(0,1,0))
                
            #derivative of Gaussian filter in the y direction
            elif filt_type == "G_Y":
                test_img = scipy.ndimage.gaussian_filter(image,sigma=(s,s,0),order=(1,0,0))
            
            responses.append(test_img)
    
    filter_responses = np.dstack(responses)
    
    return filter_responses

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * dictionary: numpy.ndarray of shape (K, 3F)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''
    
    # if single channel, stack channel 3 times
    if(len(image.shape) < 3):
        image = np.dstack((image,image,image))
    
    #convert to LAB because dictionary was calculated using LAB images
    image=skimage.color.rgb2lab(image)
    
    #create a 1x60 array for each pixel in image, and set the first 3 columns
    #of each array to be the 3-channeled values of image (image[r][c] is size 3)
    test = list()
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            temp = np.zeros((1,dictionary.shape[1]))
            temp[0,:image.shape[2]] = image[r][c]
            test.append(temp)
            
    #concatenate all the responses so that we have an array of size (image.size x dict.shape[1]),
    #where each row represents the location of a given pixel in 60-space
    test=np.vstack(test)
    
    #spatial.distance.cdist gives an array of size (num_pixels x K),
    #where d[2][5] is the distance between the 2nd pixel and the 5th K cluster.
    #if we take the index of the smallest value in each row, 
    #that corresonds to the closest K for each pixel. 
    d = scipy.spatial.distance.cdist(test,dictionary)
    wordmap = np.zeros(image.shape[0]*image.shape[1])
    for r in range(d.shape[0]):
        wordmap[r] = np.argmin(d[r])
    
    #resize the wordmap to the same size as image and show w/ fancy colormapping
    wordmap = wordmap.reshape(image.shape[0],image.shape[1])
    plt.imshow(wordmap,cmap='jet')
    plt.show()
    
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''

    #load image and run it thru the filter bank
    i, alpha, image_path = args
    image = skimage.io.imread(image_path)
    responses = extract_filter_responses(image)
    
    #reshape (H x W x 3F) array into (HW x 3F) array.
    #this is now a list of every pixel's response to the filters
    responses = responses.reshape((-1,responses.shape[2]))
    
    #get the first alpha random pixels from this new array
    sampled_response = np.random.permutation(responses)[:alpha,:]
    mypath = "../data/temp/"
    
    #create and save temporary numpy file containing filter responses for image
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    np.save(mypath+str(i),sampled_response)
    
    
    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")
    files = train_data['files']
    
    mypath = "../data/"
    temp_path = "../data/temp/"
    
    
    alpha = 75 #number of pixels per image to use
    K = 200 #number of means to use in Kmeans
    
    #placeholders for responses and args
    responses = list()
    args = list()
    import datetime
    
    #create list of args tuples for use in compute_dicitonary_one_image calls
    for i in range(files.size):
        args.append((i,alpha,mypath+files[i]))

    #run compute_dictionary_one_image for each element of args.
    #this code runs multiple processes in parallel.
    #it brought the time to compute 1000 imgs to 7 minutes down from 15
    print("starting multi @:",datetime.datetime.now())
    pool = multiprocessing.Pool(num_workers)
    pool.map(compute_dictionary_one_image,args)
    print("finishing multi @:", datetime.datetime.now())
    
    #load up all the temporary files with filter responses, and concatenate them all together
    for i in range(files.size):
        temp_data = np.load(temp_path+str(i)+".npy")
        responses.append(temp_data)
        if i == 0:
            print("loaded response size:",temp_data.shape)
        
    filter_responses = np.vstack(responses)
    print("filter_responses size:",filter_responses.shape)
    
    #run kmeans clustering on filter_responses, cluster the centers, and save the dictionary file
    print("starting kmeans@:",datetime.datetime.now())
    kmeans = sklearn.cluster.KMeans(n_clusters=K,n_jobs=num_workers).fit(filter_responses)
    print("done with cluster@:",datetime.datetime.now())
    
    print("starting cluster centers@",datetime.datetime.now())
    dictionary = kmeans.cluster_centers_
    print("done with cluster centers@:",datetime.datetime.now())
    
    np.save('dictionary.npy',dictionary)
    

    pass


