import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable
import random
from numba import jit
import numba

def pre_processing(source_image, filter_image):
    denominator = len(filter_image) * len(filter_image[0])
    terminator = len(source_image) * len(source_image[0])
    scaling = denominator / terminator
    print(f"scaling: {scaling}")
    if scaling < 1.1:
        print(f"scaling: {scaling}")
        a = len(source_image) * 1.1
        b = len(source_image[0]) * 1.1
        a = math.ceil(a)
        b = math.ceil(b)
        filter_image = tf.image.resize(filter_image, [a,b])
    return filter_image

@jit#(nopython=True)
def calculate_RMSE(input_array, comparison_array):
    diff_array1 = (int(input_array[2]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[3]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[4]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

@jit#(parallel=True)
def replace_pixel(input_list, filter_1D_array):
    output_array = np.zeros(shape=(len(input_list),1))
    for key, key_value in enumerate(input_list):
        output_array[key] = calculate_RMSE(key_value, filter_1D_array)
    result = np.where(output_array == np.amin(output_array))
    return result[0][0]


if __name__=='__main__':
    source_img_path = "C:\\VisualStudioCode\\Project3\\juanitocd01_150_225.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project3\\juanitocd170_225.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)  
    filter_image = pre_processing(source_image, filter_image)
    #
    filter_image = filter_image.numpy()
    filter_image = filter_image.astype('uint8')
    #print(filter_image)
    filter_image = filter_image.reshape(len(filter_image) * len(filter_image[0]) ,3)
    #filter_image = filter_image.tolist()
    result_array = np.zeros((len(source_image),len(source_image[0]),3), dtype=np.uint8)
    source_image = np.array(source_image) 

    input_list = numba.typed.List()
    for x_direction, key_value in enumerate(source_image):
        for y_direction, key_value2 in enumerate(key_value):
            #print(f"key:{key} value: {key_value}")
            tmp_array = np.array([x_direction, y_direction, key_value2[0], key_value2[1], key_value2[2]], dtype=int)
            input_list.append(tmp_array)

    filter_list = numba.typed.List()
    for x_direction, key_value in enumerate(filter_image):
        filter_list.append(key_value)
    



    for key, filter_list_key in enumerate(filter_list):
        #print(f"key is: {key}")
        #key = 0
        output_value = replace_pixel(input_list, filter_list_key)
        #print(input_list[output_value])
        x = input_list[output_value][0]
        y = input_list[output_value][1]
        result_array[x][y] = filter_list_key
        #print(input_list[output_value][0], )
        #print(input_list[output_value])
        #print(filter_list[key])
        input_list.pop(output_value)
        print(len(input_list))
        if len(input_list) == 0:
            break
        
        #filter_list.pop(key)
    
    #print(f"result_list is: {result_array[28]}")

    output_image = tf.convert_to_tensor(result_array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image)
    tf.io.write_file("C:\\VisualStudioCode\\Project3\\remade_image_low_resolution.jpg", output_img)
    imgplot = plt.imshow(output_image)
    plt.show()


if False:
    # source_list(x_direction, y_direction, R, G, B)
    input_array = np.array([[81, 23, 22],[255, 201, 200],[124, 66, 65]], dtype=np.uint8)
    
    
   
    filter_array = np.array([155, 97, 96], dtype=np.uint8)
    filter_list = numba.typed.List()



    #result_Array, filter_image, source_image
    print(f"result:array dimensions:{len(result_Array[0]) * len(result_Array)}")
    print(f"filter:array dimensions:{len(filter_image)}")


    for key, key_value in sorted(enumerate(result_Array),key=lambda _: random.random()):
        print(f"key:{key}")
        for key2, key_value2 in sorted(enumerate(key_value),key=lambda _: random.random()):
            key_value[key2] = replace_pixel(source_image[key][key2], filter_image)

    # for key, key_value in enumerate(result_Array):
    #     print(f"key:{key}")
    #     for key2, key_value2 in enumerate(key_value):
    #         key_value[key2] = replace_pixel(source_image[key][key2], filter_image)



    output_image = tf.convert_to_tensor(result_Array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image)
    tf.io.write_file("C:\\VisualStudioCode\\Project2\\remade_image_low_resolution.jpg", output_img)
    imgplot = plt.imshow(output_image)
    plt.show()

   