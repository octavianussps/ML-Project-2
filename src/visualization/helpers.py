
import numpy as np
import matplotlib.image as mpimg
import matplotlib as plt
import os
import re
from PIL import Image
from tensorflow.keras.utils import to_categorical

#PARAMETERS
PIXEL_DEPTH = 255
FOREGROUND_THRESHOLD = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def load_image(filename):
    """    
    input: 
    	filename
    
    output:
    	mapping image
    """
    return mpimg.imread(filename)


def load_data(data_path):
    """loading the data
    
    input: 
    	data_path: path to the data
    
    output:
    	arrays of the images
  
    """
    files = os.listdir(data_path)
    n = len(files)
    imgs = [load_image(data_path + '/' + files[i]) for i in range(n)]

    return np.asarray(imgs)



def patch_to_label(patch):
    """
    Maps a BW white patch image to a label using thresholding
    in different words : assign a label to a patch
    
    input: 
    	patch: patch to label
    
    output:
    	label
  
    """
    
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def img_crop(im, w, h, stride, padding):
    """
    Crop an image into patches, taking into account mirror boundary conditions.
    
    input: 
    	im, image
        w, 
        h, 
        stride,
        padding
    
    output:
    	list_patches 
  
    """
    
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    
    imgheight = im.shape[1]
    
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    
    for i in range(padding, imgheight + padding, stride):
        for j in range(padding, imgwidth + padding, stride):
            im_patch = im[j - padding:j + w + padding, i - padding:i + h + padding, :]
            list_patches.append(im_patch)
    
    return list_patches


def pad_image(data, padding):
    """
    Extend the canvas of an image. Mirror boundary conditions are applied.
    
    input: 
    	data
        padding
    
    output:
    	data 
  
    """
    if len(data.shape) < 3:  #we have a greyscale image (ground truth)
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    else: # the image is an RGB image
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    return data


def pad_images(images, padding_size):
    """
    Extend the canvas of an image set
    
    input: 
    	images
        padding_size
    
    output:
    	extention of the canva of images
  
    """
    nb_images = images.shape[0]
    return np.asarray([pad_image(images[i], padding_size) for i in range(nb_images)])


def img_float_to_uint8(img):
    """
    change the image codded in float to an image coded in uint8
    
    input: 
    	im: image
    
    output:
    	r_ing
    """
    r_img = img - np.min(img)
    return (r_img / np.max(r_img) * PIXEL_DEPTH).round().astype(np.uint8)


def load_images(train_data_filename, train_labels_filename, num_images):
    """
    load images
    
    input: 
    	train_data_filename: diction to the train set 
        train_labels_filename:
        num_images : nb images
    
    output:
    	images : images
        gt_images
    """
    print("loading all images from the disk ...  ")
    images = np.asarray([load_image(train_data_filename + "satImage_%.3d" % i + ".png") for i in range(1, num_images + 1)])
    gt_images = np.asarray([load_image(train_labels_filename + "satImage_%.3d" % i + ".png") for i in range(1, num_images + 1)])
    print("finished loading all images from the disk ...")
    print("images shape : ", end=' ')
    print(images.shape)
    return images, gt_images


def group_patches(patches, num_images):
    """
    input: 
    	patches: diction to the train set 
        num_images : nb images
    
    output:
    	list_windows
    """
    return patches.reshape(num_images, -1)

def image_crop_for_reg(im, window_size):
    """
    Crop an image into patches, taking into account mirror boundary conditions.
    input: 
    	patches: diction to the train set 
        num_images : nb images
    
    output:
    	list_of_windows
    """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_of_windows = []
    imagewidth = im.shape[0]
    imageheight = im.shape[1]
    
    for i in range(0, imageheight, window_size):
        for j in range(0, imagewidth, window_size):
            im_patch = im[j:j + window_size, i:i + window_size, :]
            list_of_windows.append(im_patch)
    
    return list_of_windows

def generation_windows_for_reg(images, window_size):
    """
    generation of windows for regretion
    input: 
    	images
        window_size
    
    output:
    	windows
    """                
    windows = np.asarray([image_crop_for_reg(images[i],window_size) for i in range(images.shape[0])])                    
                        
    return windows.reshape(-1, windows.shape[2], windows.shape[3], windows.shape[4])




def gen_patches(images, window_size, patch_size):
    """
    Generate patches from image
    input: 
    	images
        window_size
        patch_size
    
    output:
    	windows
    """   
    padding_size = int((window_size - patch_size) / 2)
    
    patches = np.asarray(
        [img_crop(images[i], patch_size, patch_size, patch_size, padding_size) for i in range(images.shape[0])])

    return patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])

def generate_submission(model, submission_path, modelType, *image_filenames):
    """
    Generate a .csv containing the classification of the test set
    input: 
    	model
        submission_path
        modelType
        image_filenames
    
    output:
    	classification of the test set
    """ 
    with open(submission_path, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            if modelType == 2:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model, fn))
            else:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_for_smaller_patches(model, fn))

def mask_to_submission_strings(model, image_filename):
    """
    Reads an testing image (RGB), predicts labels and outputs the strings
    input: 
    	model
        image_filename
    """
        
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    labels = model.predict(image).reshape(-1) #we predict the labels

    patch_size=16
    count = 0
   
    print("Processing the image : " + image_filename)
    for j in range(0,image.shape[2], patch_size):
        for i in range(0,image.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def mask_to_submission_strings_for_smaller_patches(model, image_filename):
    """
    Reads an testing image (RGB), predicts labels and outputs the strings for smaller patches
    input: 
    	model
        image_filename
    """
   
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  
    labels = model.predict(image).reshape(-1) #we predict the labels

    patch_size=16
    count = 0
    i_count = 0
    
    print("Processing the image : " + image_filename)

    for j in range(0,image.shape[2], patch_size):
        i_count += 456
        for i in range(0,image.shape[1], patch_size):  

            first_row = int(labels[i_count-456]+labels[i_count+1-456]+labels[i_count+2-456]+labels[i_count+3-456])
            second_row = int(labels[i_count-456+152]+labels[i_count-456+153]+labels[i_count-456+154]+labels[i_count-456+155])
            third_row = int(labels[i_count-456+304]+labels[i_count-456+305]+labels[i_count-456+306]+labels[i_count-456+307])
            fourth_row = int(labels[i_count-456+456]+labels[i_count-456+457]+labels[i_count-456+458]+labels[i_count-456+459])
                
            i_count = i_count + 4
            
            
            mean = (first_row+second_row+third_row+fourth_row) / patch_size
            if mean > 0.25:
                label = 1
            else:
                label = 0
            
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def predicted_BW(model, image_filename): 
    """
    predicted BW
    input: 
    	model
        image_filename
    
    output labels
    """

    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    return  model.predict(image).reshape(-1)#we predict the labels

