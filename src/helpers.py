import os
import numpy as np
import matplotlib.image as mpimg
import re

from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib as plt

PIXEL_DEPTH = 255
FOREGROUND_THRESHOLD = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def load_image(filename):
    return mpimg.imread(filename)


def load_data(data_path):
    files = os.listdir(data_path)
    n = len(files)
    imgs = [load_image(data_path + '/' + files[i]) for i in range(n)]

    return np.asarray(imgs)


# assign a label to a patch
def patch_to_label(patch):
    """Maps a BW white patch image to a label using thresholding """
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def img_crop(im, w, h, stride, padding):
    """ Crop an image into patches, taking into account mirror boundary conditions. """
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
    """
    if len(data.shape) < 3:
        # Greyscale image (ground truth)
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    else:
        # RGB image
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    return data


def pad_images(images, padding_size):
    """Extend the canvas of an image set"""
    nb_images = images.shape[0]
    return np.asarray([pad_image(images[i], padding_size) for i in range(nb_images)])


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def load_images(train_dir, gt_dir, num_images):
    print("loading images from disk .. ")
    imgs = np.asarray([load_image(train_dir + "satImage_%.3d" % i + ".png") for i in range(1, num_images + 1)])
    gt_imgs = np.asarray([load_image(gt_dir + "satImage_%.3d" % i + ".png") for i in range(1, num_images + 1)])
    print("finished loading images from disk ..")
    print("imgs shape : ", end=' ')
    print(imgs.shape)
    return imgs, gt_imgs


def group_patches(patches, num_images):
    return patches.reshape(num_images, -1)

def img_crop_for_reg(im, window_size):
    """ Crop an image into patches, taking into account mirror boundary conditions. """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_windows = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    
    for i in range(0, imgheight, window_size):
        for j in range(0, imgwidth, window_size):
            im_patch = im[j:j + window_size, i:i + window_size, :]
            list_windows.append(im_patch)
    
    return list_windows

def gen_windows_for_reg(imgs, window_size):
                        
    windows = np.asarray([img_crop_for_reg(imgs[i],window_size) for i in range(imgs.shape[0])])                    
                        
    return windows.reshape(-1, windows.shape[2], windows.shape[3], windows.shape[4])




def gen_patches(imgs, window_size, patch_size):
    """Generate patches from image"""
    padding_size = int((window_size - patch_size) / 2)
    
    patches = np.asarray(
        [img_crop(imgs[i], patch_size, patch_size, patch_size, padding_size) for i in range(imgs.shape[0])])

    return patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])
    #returned in (

def generate_submission(model, submission_path, modelType, *image_filenames):
    """ Generate a .csv containing the classification of the test set. """
    with open(submission_path, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            if modelType == 1:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_unet(fn))
            elif modelType == 2:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model, fn))
            else:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_for_smaller_patches(model, fn))

def mask_to_submission_strings(model, image_filename):
    """Reads an testing image (RGB), predicts labels and outputs the strings that should go into the submission file"""
    
    #assume model=cnn
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #print("loaded test image after reshape",image.shape)
    
    
    # PREDICT LABELS
    labels = model.predict(image)
    #print("labelsshape1",labels.shape)
    labels = labels.reshape(-1)
    print("labelsshape2",labels.shape)
    patch_size=16
    #patch_size = model.patch_size #patch size is an attribute
    count = 0
    #print("imgshape2",image.shape[2])
    print("Processing image => " + image_filename)
    for j in range(0,image.shape[2], patch_size):
        for i in range(0,image.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            #print("count",count)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def mask_to_submission_strings_for_smaller_patches(model, image_filename):
    """Reads an testing image (RGB), predicts labels and outputs the strings that should go into the submission file"""
    
    #assume model=cnn
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #print("loaded test image after reshape",image.shape)
    
    
    # PREDICT LABELS
    labels = model.predict(image)
    #print("labelsshape1",labels.shape)
    labels = labels.reshape(-1)
    #print("labelsshape2",labels.shape)
    patch_size=16
    #print("imageshape zeilen",image.shape[2])
    #print("imageshape spalten",image.shape[1])
    #patch_size = model.patch_size #patch size is an attribute
    count = 0
    i_count = 0
    #print("imgshape2",image.shape[2])
    print("Processing image => " + image_filename)
    for j in range(0,image.shape[2], patch_size):
        i_count += 456
        for i in range(0,image.shape[1], patch_size):  
            #print("jooot",j//16)
            firstrow = int(labels[i_count-456]+labels[i_count+1-456]+labels[i_count+2-456]+labels[i_count+3-456])
            secondrow = int(labels[i_count-456+152]+labels[i_count-456+153]+labels[i_count-456+154]+labels[i_count-456+155])
            thirdrow = int(labels[i_count-456+304]+labels[i_count-456+305]+labels[i_count-456+306]+labels[i_count-456+307])
            fourthrow = int(labels[i_count-456+456]+labels[i_count-456+457]+labels[i_count-456+458]+labels[i_count-456+459])
                
            i_count = i_count + 4
            
            
            mean = (firstrow+secondrow+thirdrow+fourthrow) / 16
            if mean > 0.25:
                label = 1
            else:
                label = 0
            
            #label = int(labels[count])
            #i_count = i_count + 2 + (j//16 * 152)
            #count += 1
            #print("count",count)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def predicted_BW(model, image_filename):
    """FOR VISUALIZATION"""

    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #print("loaded test image after reshape",image.shape)
    
    
    # PREDICT LABELS
    labels = model.predict(image)
    #print("labelsshape1",labels.shape)
    labels = labels.reshape(-1)
    #print("labelsshape2",labels.shape)
    
    return labels


def mask_to_submission_strings_unet(image_filename):
    """Reads a predicted image (BW) and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i: i + patch_size, j: j + patch_size]
            label = patch_to_label(patch)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)


def gen_image_predictions_unet(model, prediction_path, *image_filenames):
    """Predicts labels and export them as BW images"""
    for idx, path in enumerate(image_filenames):
        img = np.squeeze(load_image(path))
        prediction = img_predict_unet(img, model)
        prediction = np.squeeze(prediction).round()
        prediction = img_float_to_uint8(prediction)
        prediction_name = prediction_path + 'pred_' + str(idx + 1) + '_unet.png'
        Image.fromarray(prediction).save(prediction_name)


def img_predict_unet(img, model):
    width = img.shape[0]
    height = img.shape[1]

    img1 = img[:400, :400]
    img2 = img[:400, -400:]
    img3 = img[-400:, :400]
    img4 = img[-400:, -400:]

    imgs = np.array([img1, img2, img3, img4])
    predictions = model.predict(imgs)

    prediction = np.zeros((width, height, 1))

    prediction[:400, :400] = predictions[0]
    prediction[:400, -400:] = predictions[1]
    prediction[-400:, :400] = predictions[2]
    prediction[-400:, -400:] = predictions[3]

    return prediction


def plot_metric_history(f1_scores):
    plt.plot(f1_scores)
    plt.xlabel('# epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score for every epochs')
    plt.savefig('F1-Scores.png')
