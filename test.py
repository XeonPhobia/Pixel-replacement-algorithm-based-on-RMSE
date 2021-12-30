import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable
import random

def calculate_RMSE(input_tensor, comparison_array):
    input_array = np.array(input_tensor[0])   
    #comparison_array = np.array(comparison_tensor[0])
    #print(input_array)
    #print(comparison_array)
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

input_tensor = tf.constant([[[100, 110, 120]]], dtype=np.int8)
test_variable = np.array([[[ 81,  23,  22], [255, 201, 200]] , [[124,  66 , 65], [155,  97,  96]]], dtype=np.int8)
#print(input_tensor)
#print(list(test_variable))
#for key in range(test_variable):
#    print(key, " verdi: ", test_variable[key])
#output_array = np.array([], dtype=np.int8)
output_array = np.zeros(shape=(len(test_variable),len(test_variable[0]),1))
#output_array = map(calculate_RMSE,input_tensor, test_variable)
#print(output_array)
#variabel2 = map(calculate_RMSE(input_tensor,test_variable), test_variable)
#a = 196609
for x_direction in range(len(test_variable)):
    for y_direction in range(len(test_variable[x_direction])):
        #print(x_direction, y_direction, test_variable[x_direction][y_direction])
        output_array[x_direction][y_direction] = calculate_RMSE(input_tensor[0], test_variable[x_direction][y_direction])


print(output_array)
print(np.amin(output_array))
result = np.where(output_array == np.amin(output_array))
print(result[0])
print(result[1])