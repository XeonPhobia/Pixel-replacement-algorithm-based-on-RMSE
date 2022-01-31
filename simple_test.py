from pickletools import uint8
#from numba import jit
import numba
import numpy as np
import cupy as cp
import math

# @jit
# def add1(x):
#     return x + 1

# @jit
# def add1array(input):
#     for key in range(len(input)):
#         input[key] = 3
#     return input

# input_array = np.array([[81, 23, 22], [255, 201, 200], [124, 66, 65], [155, 97, 96]], dtype=np.uint8)
# input_list = numba.typed.List()
# input_list.append(input_array)
# print(input_list[0][0])
# input_list[0][0].pop()
#AttributeError: 'numpy.ndarray' object has no attribute 'pop'

def calculate_RMSE(input_array, comparison_array):
    #input_array = np.array(input_tensor)   
    diff_array1 = (int(input_array[2]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[3]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[4]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

# def replace_pixel(input_tensor, test_variable):
#     output_array = np.zeros(shape=(len(test_variable),1))
#     for key, key_value in enumerate(test_variable):
#         output_array[key] = calculate_RMSE(input_tensor, key_value)
#     result = np.where(output_array == np.amin(output_array))
#     return test_variable.pop(result[0][0])

input_array = np.array([0, 0, 81, 23, 22], dtype=np.uint8)
input_array2 = np.array([0, 1, 255, 201, 200], dtype=np.uint8)
input_array3 = np.array([1, 0, 124, 66, 65], dtype=np.uint8)
filter_array = np.array([155, 97, 96], dtype=np.uint8)
input_list = numba.typed.List()
input_list.append(input_array)
input_list.append(input_array2)
input_list.append(input_array3)


print(calculate_RMSE(input_list[0], filter_array))







# source_list(x_direction, y_direction, R, G, B)
#source_list = numba.typed.List([0, 0, 255, 250, 240])
#source_list = source_list.append([1, 0, 81, 23, 22])
#filter_list(R, G, B, darkness(R + G + B))
#filter_list = cp.array([[81, 23, 22], [255, 201, 200], [124, 66, 65], [155, 97, 96]], dtype=uint8)
#print(source_list)


#print(filter_list)

#print(add1array(input_list))


