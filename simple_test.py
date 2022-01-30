from pickletools import uint8
#from numba import jit
import numba
import numpy as np
import cupy as cp


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


input_array = np.array([81, 23, 22], dtype=np.uint8)
input_array2 = np.array([255, 201, 200], dtype=np.uint8)
input_list = numba.typed.List()
input_list.append(input_array)
input_list.append(input_array2)
print(input_list[0])
input_list.pop(1)
print(input_list[0])






# source_list(x_direction, y_direction, R, G, B)
#source_list = numba.typed.List([0, 0, 255, 250, 240])
#source_list = source_list.append([1, 0, 81, 23, 22])
#filter_list(R, G, B, darkness(R + G + B))
#filter_list = cp.array([[81, 23, 22], [255, 201, 200], [124, 66, 65], [155, 97, 96]], dtype=uint8)
#print(source_list)


#print(filter_list)

#print(add1array(input_list))


