import os
from pickletools import uint8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

def calculate_RMSE():
    rmse_difference = tf.math.subtract(input_tensor, filter_tensor)
    rmse_difference = tf.math.square(rmse_difference)
    rmse_difference = tf.reduce_sum(rmse_difference, -1)
    rmse_difference = tf.math.sqrt(rmse_difference)
    tmp_variable = tf.reduce_sum(rmse_difference)
    return tmp_variable

def swap_two_indices(var1, var2, score, filter_tensor):
    var1 = tf.cast(var1, tf.uint8)
    var2 = tf.cast(var2, tf.uint8)
    indices[var1[0]][var1[1]], indices[var2[0]][var2[1]] = indices[var2[0]][var2[1]], indices[var1[0]][var1[1]]
    filter_tensor_temp = tf.scatter_nd(indices, filter_tensor, filter_tensor.shape)
    if calculate_RMSE() < score:
        filter_tensor = filter_tensor_temp
        score = calculate_RMSE()
    return score, filter_tensor

if __name__=='__main__':
    #Convert two .jpg images to tensors with R G B values between 0 and 255:
    input_tensor = tf.constant([[[81, 23, 22],[255, 201, 200]],[[124, 66, 65],[0, 0, 0]]], dtype=np.float32)
    filter_tensor = tf.Variable([[[35, 132, 10],[137, 184, 30]],[[66, 12, 254],[0, 0, 0]]], dtype=np.float32)

    indices = np.zeros((filter_tensor.shape[0],filter_tensor.shape[1],2), dtype=np.int32)
    for key_x in range(len(indices)):
        for key_y in range(len(indices[0])):
            indices[key_x][key_y] = [key_x, key_y]
    #indices = tf.Variable(indices, dtype=np.int32)

    rmse_difference = tf.zeros([input_tensor.shape[0], input_tensor.shape[1], 3], dtype=np.float32)
    # var1 = tf.Variable([0, 0], dtype=np.uint8)
    # var2 = tf.Variable([1, 0], dtype=np.uint8)
    var1 = tf.Variable([0, 0], dtype=np.float32)
    var2 = tf.Variable([1, 0], dtype=np.float32)
    score = calculate_RMSE()
    y = score

    # I would like to write an optimizer in tensorflow that reorders [R, G, B] array in filter_tensor variable.
    # The goal is to change the indices order of filter_tensor in order for calculate_RMSE() to be as small as possible. 

    # indices, loss_function = loss_value
    # tf.constant([[[0,0], [0,1]], [[1,0], [1,1]]]), calculate_RMSE() = 531.4941 <= initial value/score
    # tf.constant([[[0,0], [1,0]], [[0,1], [1,1]]]), calculate_RMSE() = 515.36847 <= this is optimal solution. 
    # tf.constant([[[1, 0],[0.0]],[[1, 0],[1, 1]]]), calculate_RMSE() = 655.2762
    # tf.constant([[[0, 1],[0.0]],[[1, 0],[1, 1]]]), calculate_RMSE() = 674.356

    with tf.GradientTape(persistent=True) as tape:
        # tensorflow gradientTape sources 
        # var1 only whole numbers between 0 and filter_tensor.shape[0]
        # var2 only whole numbers between 0 and filter_tensor.shape[1]
        y, filter_tensor = swap_two_indices(var1, var2, score, filter_tensor)

    dy_dswap_two_indices = tape.gradient(y, [var1, var2])

    print(dy_dswap_two_indices)