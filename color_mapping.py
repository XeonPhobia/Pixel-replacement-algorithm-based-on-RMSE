import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable
import random
from joblib import Parallel, delayed

def pre_processing(source_image, filter_image):
    denominator = len(filter_image) * len(filter_image[0])
    terminator = len(source_image) * len(source_image[0])
    scaling = denominator / terminator
    print(f"scaling: {scaling}")
    if scaling < 1.02:
        print(f"scaling: {scaling}")
        a = len(source_image) * 1.05
        b = len(source_image[0]) * 1.05
        a = math.ceil(a)
        b = math.ceil(b)
        filter_image = tf.image.resize(filter_image, [a,b])
    return filter_image

def calculate_RMSE(input_tensor, comparison_array):
    input_array = np.array(input_tensor)   
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

def replace_pixel(input_tensor, test_variable):
    output_array = np.zeros(shape=(len(test_variable),1))
    for key, key_value in enumerate(test_variable):
        output_array[key] = calculate_RMSE(input_tensor, key_value)
    result = np.where(output_array == np.amin(output_array))
    return test_variable.pop(result[0][0])

if __name__=='__main__':
    source_img_path = "C:\\VisualStudioCode\\Project2\\juanitocd01_150_225.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project2\\Julius_low_res.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)
    #koblingstabell_dictionary = {}
    filter_image = pre_processing(source_image, filter_image)
    filter_image = filter_image.numpy()
    
    filter_image = filter_image.reshape(len(filter_image) * len(filter_image[0]) ,3)
    filter_image = filter_image.tolist()
    result_Array = np.zeros((len(source_image),len(source_image[0]),3), dtype=np.uint8)
    #result_Array, filter_image, source_image
    print(f"result:array dimensions:{len(result_Array), len(result_Array[0])}")
    print(f"filter:array dimensions:{len(filter_image)}")

    for key, key_value in enumerate(result_Array):
        print(f"key:{key}")
        for key2, key_value2 in enumerate(key_value):
            key_value[key2] = replace_pixel(source_image[key][key2], filter_image)

    output_image = tf.convert_to_tensor(result_Array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image)
    #print(filter_image)git

    tf.io.write_file("C:\\VisualStudioCode\\Project2\\remade_image_low_resolution.jpg", output_img)
    #source_image=tf.image.convert_image_dtype(source_image, dtype=tf.float32) 
    imgplot = plt.imshow(output_image)
    plt.show()

   