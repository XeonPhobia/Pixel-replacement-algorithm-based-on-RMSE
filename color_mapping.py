import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable
import random

def calculate_RMSE(input_tensor, comparison_tensor):
    input_array = np.array(input_tensor[0])   
    comparison_array = np.array(comparison_tensor[0])
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

def replace_pixel(source_image, filter_image, source_pixel, koblingstabell_dictionary):
    temp_dictionary = {}
    length_filter_image = tf.Tensor.get_shape(filter_image).as_list()
    length_filter_image = length_filter_image[0]
    for key in range(length_filter_image):
        temp_dictionary[key] = calculate_RMSE(source_image[source_pixel], filter_image[key])
    for key2 in koblingstabell_dictionary: 
        temp_dictionary.pop(koblingstabell_dictionary[key2])
    best_fit = min(temp_dictionary, key=temp_dictionary.get)
    return best_fit



if __name__=='__main__':
    source_img_path = "C:\\VisualStudioCode\\Project2\\redblue.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project2\\Burgunder gul blå rosa.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)
    koblingstabell_dictionary = {}
    #koblingstabell_dictionary[0] = replace_pixel(source_image, filter_image, 0, koblingstabell_dictionary)  
    #koblingstabell_dictionary[1] = replace_pixel(source_image, filter_image, 1, koblingstabell_dictionary)
    filter_image = np.array(filter_image)
    print("zeroth",filter_image[0])
    print("first",filter_image[1])
    #print(filter_image[0])

    if False:
        #print(fit_dictionary[0], fit_dictionary[100])
        print(fit_dictionary.get(0))
        print(fit_dictionary.get(1))
        print(min(fit_dictionary, key=fit_dictionary.get)) #finn filter_image posisjonen/pixelet som er likest. 
        print(filter_image[min(fit_dictionary, key=fit_dictionary.get)])
        #source_image[0] = filter_image[min(fit_dictionary, key=fit_dictionary.get)]
        print("New source image with one pixel from filter image is:")
        print(source_image)
        output_image = tf.convert_to_tensor(filter_image[min(fit_dictionary, key=fit_dictionary.get)], dtype='uint8')
        #print(output_image)
        #if False:
        output_image = tf.reshape(output_image, [1, 1, 3])
        #print(a)
        output_img=tf.image.encode_jpeg(output_image)
        #tf.io.write_file("C:\\VisualStudioCode\\Project2\\ooutputredblue.jpg", output_img)
        source_image=tf.image.convert_image_dtype(source_image, dtype=tf.float32) 
        imgplot = plt.imshow(output_image)
        plt.show()