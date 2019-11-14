import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage
import skimage.io

if __name__ == '__main__':
    num_cores = util.get_num_CPU()

    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)

    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    visual_words.compute_dictionary(num_workers=num_cores)

    dictionary = np.load('dictionary.npy')
    filename = "test.jpg"
    img = visual_words.get_visual_words(image,dictionary)
    util.save_wordmap(img, filename)
    visual_recog.build_recognition_system(num_workers=num_cores)

    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    deep_recog.build_recognition_system(vgg16,num_workers=num_cores)
    conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

