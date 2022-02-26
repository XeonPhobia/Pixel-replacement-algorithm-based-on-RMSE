import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from numba import jit
import numba
import numpy as np
import math
import tensorflow as tf


def calculate_RMSE():
    rmse_difference = tf.math.subtract(input_tensor, filter_tensor)
    rmse_difference = tf.math.square(rmse_difference)
    rmse_difference = tf.reduce_sum(rmse_difference, -1)
    rmse_difference = tf.cast(rmse_difference, dtype=np.float32)
    rmse_difference = tf.math.sqrt(rmse_difference)
    return tf.reduce_sum(rmse_difference)



if __name__=='__main__':
    input_tensor = tf.constant([[[81, 23, 22],[255, 201, 200]],[[124, 66, 65],[0, 0, 0]]], dtype=np.uint8)
    filter_tensor = tf.Variable([[[35, 132, 10],[137, 184, 30]],[[66, 12, 254],[0, 0, 0]]], dtype=np.uint8)
    input_tensor = tf.cast(input_tensor, dtype=np.int32)
    total_Score = tf.zeros(1, dtype=np.float32)
    
    filter_tensor = tf.cast(filter_tensor, dtype=np.int32)
    rmse_difference = tf.zeros([input_tensor.shape[0], input_tensor.shape[1], 3], dtype=np.int32)
    
    total_Score = calculate_RMSE()

    # I would like to write an optimizer in tensorflow that reorders [R, G, B] array in filter_tensor variable.
    # The goal is to get total_Score to be as small as possible. 

    

    print(total_Score)