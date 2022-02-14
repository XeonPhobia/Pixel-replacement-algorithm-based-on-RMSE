import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import numba
import random
import cProfile
import timeit

def print_length():
    if filter_list:
        print(f"length of filter_list {len(filter_list)}")
    if List_RlGlBl:
        print(f"length of List_RlGlBl {len(List_RlGlBl)}")
    if List_RhGlBl:    
        print(f"length of List_RhGlBl {len(List_RhGlBl)}")
    if List_RlGhBl:    
        print(f"length of List_RlGhBl {len(List_RlGhBl)}")
    if List_RlGlBh:    
        print(f"length of List_RlGlBh {len(List_RlGlBh)}")
    if List_RhGhBl:    
        print(f"length of List_RhGhBl {len(List_RhGhBl)}")
    if List_RhGlBh:    
        print(f"length of List_RhGlBh {len(List_RhGlBh)}")
    if List_RlGhBh:    
        print(f"length of List_RlGhBh {len(List_RlGhBh)}")
    if List_RhGhBh:    
        print(f"length of List_RhGhBh {len(List_RhGhBh)}")
    if initializing_list:   
        print(f"length of initializing_list {len(initializing_list)}")
    if result_list:    
        print(f"length of result_list {len(result_list)}")
    
def pre_processing(source_image, filter_image):
    denominator = len(filter_image) * len(filter_image[0])
    terminator = len(source_image) * len(source_image[0])
    scaling = denominator / terminator
    if scaling < 1.1:
        print(f"scaling: {scaling}")
        a = len(source_image) * 1.1
        b = len(source_image[0]) * 1.1
        a = math.ceil(a)
        b = math.ceil(b)
        filter_image = tf.image.resize(filter_image, [a,b])
    return filter_image

@numba.njit
def calculate_RMSE(input_array, comparison_array):
    #input_array = np.array(input_tensor)   
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array


@numba.njit#(parallel=True)
def parallel_for_loop(input_coordinates, pixel_list): 
    output_array = np.zeros(shape=(len(pixel_list),1))
    input1 = source_image[input_coordinates[0], input_coordinates[1]]
    #for key3 in numba.prange(len(pixel_list)):
    for key3 in range(len(pixel_list)):    
        input2 = pixel_list[key3]
        diff_array1 = (int(input1[0]) - int(input2[0]))**2 
        diff_array2 = (int(input1[1]) - int(input2[1]))**2 
        diff_array3 = (int(input1[2]) - int(input2[2]))**2      
        diff_array = diff_array1 + diff_array2 + diff_array3 
        output_array[key3] = math.sqrt(diff_array)
    return output_array

def replace_pixel_initial(input_coordinates, pixel_list):
    output_array = parallel_for_loop(input_coordinates, pixel_list)
    result = np.where(output_array == np.amin(output_array))
    result2 = pixel_list.pop(result[0][0])
    tmp_array = np.array([input_coordinates[0], input_coordinates[1], result2[0],result2[1], result2[2]], dtype=int)
    result_list.append(tmp_array)

if __name__=='__main__':
    start = timeit.default_timer()
    source_img_path = "C:\\VisualStudioCode\\Project3\\juanitocd170_225.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project3\\juanitocd01_200_300.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)
    filter_image = pre_processing(source_image, filter_image)
    filter_image = filter_image.numpy()
    filter_image = filter_image.reshape(len(filter_image) * len(filter_image[0]) ,3)
    filter_image = filter_image.tolist()
    result_Array = np.zeros((len(source_image),len(source_image[0]),3), dtype=np.uint8)
    source_image = np.array(source_image) 

    #result_Array, filter_image, source_image
    List_RlGlBl = numba.typed.List()
    List_RhGlBl = numba.typed.List()
    List_RlGhBl = numba.typed.List()
    List_RlGlBh = numba.typed.List()
    List_RhGhBl = numba.typed.List()
    List_RhGlBh = numba.typed.List()
    List_RlGhBh = numba.typed.List()
    List_RhGhBh = numba.typed.List()
    for key2, key_value2 in enumerate(filter_image):
        key_value3 = np.array(key_value2)
        if key_value2[0] < 128:
            if key_value2[1] < 128:
                if key_value2[2] < 128:
                    List_RlGlBl.append(key_value3)
                else: 
                    List_RlGlBh.append(key_value3)
            else:
                if key_value2[2] < 128:
                    List_RlGhBl.append(key_value3)
                else: 
                    List_RlGhBh.append(key_value3)   
        else:
            if key_value2[1] < 128:
                if key_value2[2] < 128:
                    List_RhGlBl.append(key_value3)
                else: 
                    List_RhGlBh.append(key_value3)
            else:
                if key_value2[2] < 128:
                    List_RhGhBl.append(key_value3)
                else: 
                    List_RhGhBh.append(key_value3)               


    initializing_list = numba.typed.List()
    for x_direction in range(len(source_image)):
        for y_direction in range(len(source_image[0])):
            tmp_array1 = np.array([x_direction, y_direction], dtype=int)
            initializing_list.append(tmp_array1)

    result_list = numba.typed.List()
    filter_list = numba.typed.List()

    print("first fill")
    for key, key_val in reversed(list(enumerate(initializing_list))):
        if key%1000 == 0:
            print(key)
        source_pixel = source_image[key_val[0]][key_val[1]]
        if source_pixel[0] < 128:
            if source_pixel[1] < 128:
                if source_pixel[2] < 128:
                    if List_RlGlBl:
                        replace_pixel_initial(initializing_list[key], List_RlGlBl)
                        initializing_list.pop(key)
                else: 
                    if List_RlGlBh:
                        replace_pixel_initial(initializing_list[key], List_RlGlBh)
                        initializing_list.pop(key)
            else:
                if source_pixel[2] < 128:
                    if List_RlGhBl:
                        replace_pixel_initial(initializing_list[key], List_RlGhBl)
                        initializing_list.pop(key)
                else:  
                    if List_RlGhBh:
                        replace_pixel_initial(initializing_list[key], List_RlGhBh)
                        initializing_list.pop(key)
        else:
            if source_pixel[1] < 128:
                if source_pixel[2] < 128:
                    if List_RhGlBl:
                        replace_pixel_initial(initializing_list[key], List_RhGlBl)
                        initializing_list.pop(key)
                else: 
                    if List_RhGlBh:
                        replace_pixel_initial(initializing_list[key], List_RhGlBh)
                        initializing_list.pop(key)
            else:
                if source_pixel[2] < 128:
                    if List_RhGhBl:
                        replace_pixel_initial(initializing_list[key], List_RhGhBl)
                        initializing_list.pop(key)
                else: 
                    if List_RhGhBh:
                        replace_pixel_initial(initializing_list[key], List_RhGhBh)
                        initializing_list.pop(key)

        #replace_pixel_initial(initializing_list[0], List_RlGlBl)
        #initializing_list.pop(key)

    #print_length()

    if List_RlGlBl:
        for key, key_value in reversed(list(enumerate(List_RlGlBl))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RlGlBl.pop(key)
    if List_RlGlBh:
        for key, key_value in reversed(list(enumerate(List_RlGlBh))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RlGlBh.pop(key)
    if List_RlGhBl:
        for key, key_value in reversed(list(enumerate(List_RlGhBl))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RlGhBl.pop(key)
    if List_RlGhBh:
        for key, key_value in reversed(list(enumerate(List_RlGhBh))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RlGhBh.pop(key)
    if List_RhGlBl:
        for key, key_value in reversed(list(enumerate(List_RhGlBl))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RhGlBl.pop(key)        
    if List_RhGlBh:
        for key, key_value in reversed(list(enumerate(List_RhGlBh))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RhGlBh.pop(key)
    if List_RhGhBl:
        for key, key_value in reversed(list(enumerate(List_RhGhBl))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RhGhBl.pop(key)
    if List_RhGhBh:
        for key, key_value in reversed(list(enumerate(List_RhGhBh))):
            tmp_array = np.array(key_value)
            filter_list.append(tmp_array)
            List_RhGhBh.pop(key)

    
    print("refill")
    for key, key_val in reversed(list(enumerate(initializing_list))):
        if key%1000 == 0:
            print(key)
        replace_pixel_initial(key_val, filter_list)
        initializing_list.pop(key)



    for key, key_val in enumerate(result_list):
        result_Array[key_val[0]][key_val[1]] = key_val[2:5]

    stop = timeit.default_timer()
    print('Time: ', stop - start)      

    # print_length()

    output_image = tf.convert_to_tensor(result_Array, dtype='uint8')
    output_img=tf.image.encode_jpeg(output_image)
    tf.io.write_file("C:\\VisualStudioCode\\Project2\\remade_image_low_resolution.jpg", output_img)
    imgplot = plt.imshow(output_image)
    plt.show()


    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(15)