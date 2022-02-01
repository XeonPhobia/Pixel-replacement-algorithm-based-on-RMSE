from numba import jit
import numba
import numpy as np
import math

@jit#(nopython=True)
def calculate_RMSE(input_array, comparison_array):
    diff_array1 = (int(input_array[2]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[3]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[4]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

@jit
def replace_pixel(input_list, filter_1D_array):
    output_array = np.zeros(shape=(len(input_list),1))
    for key, key_value in enumerate(input_list):
        output_array[key] = calculate_RMSE(key_value, filter_1D_array)
    result = np.where(output_array == np.amin(output_array))
    return result[0][0]

if __name__=='__main__':
    # source_list(x_direction, y_direction, R, G, B)
    input_array = np.array([[81, 23, 22],[255, 201, 200],[124, 66, 65]], dtype=np.uint8)
    input_list = numba.typed.List()


    filter_array = np.array([155, 97, 96], dtype=np.uint8)
    filter_list = numba.typed.List()
    
    for key, key_value in enumerate(input_array):
        print(f"key:{key} value: {key_value}")
        input_list.append([0, key, key_value[0], key_value[1], key_value[2]], dtype=int)
    #     for key2, key_value2 in enumerate(key_value):
    #         key_value[key2] = replace_pixel(source_image[key][key2], filter_image)
    print(input_list)



    
    # result_array = np.zeros((3,1,3), dtype=np.uint8)

    # key = 0
    # output_value = replace_pixel(input_list, filter_list[key])
    # result_array[input_list[output_value][0]][input_list[output_value][1]] = input_list[output_value][2:5]
    # input_list.pop(output_value)
    # filter_list.pop(key)
    # #print(f"result_list is: {result_array}")

    