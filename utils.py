import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf
import time, datetime
import setting
from scipy.misc import imread
import sys
from PIL import Image
import math
import matplotlib.cm as cm

printstyle = {
"INFO":'\033[95m',
"TRAINING_STATUS":'\033[94m',
"VALIDATION_STATUS":'\033[92m',
"WARNING":'\033[93m',
"ERROR":'\033[91m',
"ENDC":'\033[0m',
"BOLD":'\033[1m',
"UNDERLINE":'\033[4m'
}


def get_label_info():

    config = setting.config
    class_names = []
    label_values = []
    for class_name, label_value in config["data"]["classes"].items():
        class_names.append(class_name)
        label_values.append([int(label_value[0]), int(label_value[1]), int(label_value[2])])

    return class_names, label_values

def compute_custom_class_weights(labels_dir):
    '''
    The custom class weighing function as seen in the ENet paper.
    '''
    # #initialize dictionary with all 0
    # label_to_frequency = {}
    # for i in xrange(num_classes):
    #     label_to_frequency[i] = 0

    # for n in xrange(len(image_files)):
    #     image = imread(image_files[n])

    #     #For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
    #     for i in xrange(num_classes):
    #         class_mask = np.equal(image, i)
    #         class_mask = class_mask.astype(np.float32)
    #         class_frequency = np.sum(class_mask)

    #         label_to_frequency[i] += class_frequency

    # #perform the weighing function label-wise and append the label's class weights to class_weights
    # class_weights = []
    # total_frequency = sum(label_to_frequency.values())
    # for label, frequency in label_to_frequency.items():
    #     class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
    #     class_weights.append(class_weight)

    # #Set the last class_weight to 0.0
    # class_weights[-1] = 0.0

    # return class_weights

    print_log("Computing custom class weights from the training samples", "INFO")
    print("-"*50)
    print("-"*50)

    label_values = setting.label_values

    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = setting.num_classes

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    class_weights = []
    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = 1/np.log(1.02+(class_pixels/total_pixels))

    print("\n")
    print_log("Computing custom class weights for {} label images is over".format(len(image_files)), "INFO")
    for i in range(len(setting.class_names)):
        print_log(setting.class_names[i]+" ==> "+str(class_weights[i]),"INFO")

    return class_weights


def compute_median_class_weights(labels_dir):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.
    '''
    # #Initialize all the labels key with a list value
    # label_to_frequency_dict = {}
    # for i in xrange(num_classes):
    #     label_to_frequency_dict[i] = []

    # for n in xrange(len(image_files)):
    #     image = imread(image_files[n])

    #     #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
    #     for i in xrange(num_classes):
    #         class_mask = np.equal(image, i)
    #         class_mask = class_mask.astype(np.float32)
    #         class_frequency = np.sum(class_mask)

    #         if class_frequency != 0.0:
    #             label_to_frequency_dict[i].append(class_frequency)

    # class_weights = []

    # #Get the total pixels to calculate total_frequency later
    # total_pixels = 0
    # for frequencies in label_to_frequency_dict.values():
    #     total_pixels += sum(frequencies)

    # for i, j in label_to_frequency_dict.items():
    #     j = sorted(j) #To obtain the median, we got to sort the frequencies

    #     median_frequency = np.median(j) / sum(j)
    #     total_frequency = sum(j) / total_pixels
    #     median_frequency_balanced = median_frequency / total_frequency
    #     class_weights.append(median_frequency_balanced)

    # #Set the last class_weight to 0.0 as it's the background class
    # class_weights[-1] = 0.0

    # return class_weights

    print_log("Computing median class weights from the training samples", "INFO")
    print("-"*50)
    print("-"*50)

    label_values = setting.label_values

    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = setting.num_classes

    class_weights = []
    class_pixels = [[] for i in range(num_classes)]

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_frequency = np.sum(class_map)

            if class_frequency != 0.0:
                class_pixels[index].append(class_frequency)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    for class_frequencies in class_pixels:
        total_pixels += sum(class_frequencies)

    for class_frequencies in class_pixels:
        sorted_class_frequencies = sorted(class_frequencies)
        median_frequency = np.median(sorted_class_frequencies)/sum(sorted_class_frequencies)
        total_frequency = sum(sorted_class_frequencies)/total_pixels
        median_frequency_balanced = median_frequency/total_frequency
        class_weights.append(median_frequency_balanced)

    class_weights=np.array(class_weights)

    print("\n")
    print_log("Computing median class weights for {} label images is over".format(len(image_files)), "INFO")
    for i in range(len(setting.class_names)):
        print_log(setting.class_names[i]+" ==> "+str(class_weights[i]),"INFO")

    return class_weights

def compute_default_class_weights(labels_dir):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    print_log("Computing default class weights from the training samples", "INFO")
    print("-"*50)
    print("-"*50)

    label_values = setting.label_values

    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = setting.num_classes

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    print("\n")
    print_log("Computing default class weights for {} label images is over".format(len(image_files)), "INFO")
    for i in range(len(setting.class_names)):
        print_log(setting.class_names[i]+" ==> "+str(class_weights[i]),"INFO")



    return class_weights


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # label_values = get_label_info()
    # label_values = label_values.tolist()
    
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x


def color_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def calculate_optimal_resize_image_resolution():

    config = setting.config
    w_list = []
    h_list = []
    data_dir = os.path.join(config["data"]["directory"],"train")
    for _file in sorted(os.listdir(data_dir)):
        im = Image.open(os.path.join(data_dir, _file))
        width, height = im.size

        base = int(math.fmod(math.ceil(height/8), 2))
        new_height = height
        if base>0:
            new_height = (math.ceil(height/8) - base)*8
            
        base = int(math.fmod(math.ceil(width/8), 2))
        new_width = width
        if base>0:
            new_width = (math.ceil(width/8) - base)*8
            
        h_list.append(new_height)
        w_list.append(new_width)
        

    #logic only for adapnet
    min_h = min(h_list)
    min_w = min(w_list)
   

    return [min_h, min_w]


def print_log(message=None, type=None, f=None):

    print(printstyle[type]+message+printstyle["ENDC"])
    # print(f)
    if f is not None:
        with open(f, "a+") as file:
            file.write(message+"\n")



def make_confidence_overlay(image, output_confidence):

    cmap = {0:"PiYG",
            1:"PRGn",
            2:"BrBG",
            3:"PuOr",
            4:"RdGy",
            5:"RdBu",
            6:"RdYlBu",
            7:"RdYlGn",
            8:"Spectral",
            9:"coolwarm",
            10:"bwr",
            11:"seismic"
            }

    for class_index in range(output_confidence.shape[-1]):
        single_class_output_confidence = output_confidence[:,:,class_index]
        # mycm = cm.get_cmap(cmap[class_index])
        mycm = cm.get_cmap(cmap[10])
        overimage = mycm(single_class_output_confidence, bytes=True)
        output = 0.6*overimage[:,:,0:3] + 0.4*image

    return output

def make_overlay(background_image, overlay_image):
    background = Image.fromarray(background_image)
    overlay = Image.fromarray(overlay_image)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    output = Image.blend(background, overlay, 0.5)
    # output.save("new.png","PNG")
    return output


def fast_overlay(input_image, segmentation, color=[0, 255, 0, 90]):

    color = np.array(color).reshape(1, 4)
    shape = input_image.shape
    segmentation = segmentation.reshape(shape[0], shape[1], 1)
    output = np.dot(segmentation, color)
    output = scipy.misc.toimage(output, mode="RGBA")
    background = scipy.misc.toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)
    