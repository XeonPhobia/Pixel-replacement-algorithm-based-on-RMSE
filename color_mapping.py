import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable
from numba import njit, prange
import numba
import cProfile
import timeit

def pre_processing(source_image, filter_image):
    denominator = len(filter_image) * len(filter_image[0])
    terminator = len(source_image) * len(source_image[0])
    scaling = denominator / terminator
    if scaling < 1.1:
        print(f"scaling up filter_image from: {scaling}")
        a = len(source_image) * 1.1
        b = len(source_image[0]) * 1.1
        a = math.ceil(a)
        b = math.ceil(b)
        filter_image = tf.image.resize(filter_image, [a,b])
    return filter_image

def score(source_array, target_array):
    output_array = np.zeros((len(source_array),len(source_array[0]),1), dtype=np.uint8)
    score = 0
    for x in range(len(source_array)):
        for y in range(len(source_array[0])):
            output_array[x][y] = calculate_RMSE(source_array[x][y], target_array[x][y])
            score += int(output_array[x][y])
    output_image = tf.convert_to_tensor(output_array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image, format='grayscale')
    tf.io.write_file("C:\\VisualStudioCode\\Project3\\error_map.jpg", output_img)
    print(f"score is: {score}")
    return score


def calculate_RMSE(input_array, comparison_array):
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

@njit(parallel=True)
def for_loop_replace_pixel(output_array, input_list, filter_1D_array):
    filter_1D_array_0 = int(filter_1D_array[0])
    filter_1D_array_1 = int(filter_1D_array[1])
    filter_1D_array_2 = int(filter_1D_array[2])
    for key in prange(len(input_list)):
        key_value = input_list[key]
        diff_array1 = (int(key_value[2]) - filter_1D_array_0)**2 
        diff_array2 = (int(key_value[3]) - filter_1D_array_1)**2 
        diff_array3 = (int(key_value[4]) - filter_1D_array_2)**2      
        output_array[key] = math.sqrt(diff_array1 + diff_array2 + diff_array3)
    return output_array

def main():
    source_img_path = "C:\\VisualStudioCode\\Project3\\Flo_75_112.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project3\\juanitocd170_225.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)  
    filter_image = pre_processing(source_image, filter_image)
    filter_image = filter_image.numpy()
    filter_image = filter_image.astype('uint8')
    filter_image = filter_image.reshape(len(filter_image) * len(filter_image[0]) ,3)
    result_array = np.zeros((len(source_image),len(source_image[0]),3), dtype=np.uint8)
    source_image = np.array(source_image) 
    #array in order to create a score. 
    source_comparison_image = source_image

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
        output_array = np.zeros(shape=(len(input_list),1))
        output_array = for_loop_replace_pixel(output_array, input_list, filter_list_key)
        result = np.where(output_array == np.amin(output_array))
        output_value = result[0][0]

        #output_value = replace_pixel(input_list, filter_list_key)
        x = input_list[output_value][0]
        y = input_list[output_value][1]
        result_array[x][y] = filter_list_key
        input_list.pop(output_value)
        if len(input_list) % 1000 == 0:
            print(len(input_list))
        if len(input_list) == 0:
            break
        

    score(source_comparison_image, result_array)




    output_image = tf.convert_to_tensor(result_array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image, chroma_downsampling=False)
    tf.io.write_file("C:\\VisualStudioCode\\Project3\\remade_image_low_resolution.jpg", output_img)
    #imgplot = plt.imshow(output_image)
    #plt.show()

if __name__=='__main__':
    #start = timeit.default_timer()
    main()
    #stop = timeit.default_timer()
    #print('Time: ', stop - start) 
    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(15)


