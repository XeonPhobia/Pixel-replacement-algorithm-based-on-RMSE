import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

def calculate_RMSE():
    rmse_difference = tf.math.subtract(input_tensor, filter_tensor)
    rmse_difference = tf.math.square(rmse_difference)
    rmse_difference = tf.reduce_sum(rmse_difference, -1)
    rmse_difference = tf.math.sqrt(rmse_difference)
    return tf.reduce_sum(rmse_difference)

if __name__=='__main__':
    #Convert two .jpg images to tensors with R G B values between 0 and 255:
    input_tensor = tf.constant([[[81, 23, 22],[255, 201, 200]],[[124, 66, 65],[0, 0, 0]]], dtype=np.float32)
    filter_tensor = tf.Variable([[[35, 132, 10],[137, 184, 30]],[[66, 12, 254],[0, 0, 0]]], dtype=np.float32)

    index_filter_tensor = np.zeros((filter_tensor.shape[0],filter_tensor.shape[1],2), dtype=np.int32)
    for key_x in range(len(index_filter_tensor)):
        for key_y in range(len(index_filter_tensor[0])):
            index_filter_tensor[key_x][key_y] = [key_x, key_y]

    index_filter_tensor = tf.convert_to_tensor(index_filter_tensor, dtype=np.int32)
    print(index_filter_tensor)

    #Heatmap of error. 
    rmse_difference = tf.zeros([input_tensor.shape[0], input_tensor.shape[1], 3], dtype=np.float32)

    # I would like to write an optimizer in tensorflow that reorders [R, G, B] array in filter_tensor variable.
    # The goal is to change "indices" in order for calculate_RMSE() to be as small as possible. 

    # indices, loss = loss_value
    # tf.constant([[[0,0], [0,1]], [[1,0], [1,1]]]), calculate_RMSE() = 531.4941 <= initial value/score
    # tf.constant([[[0,0], [1,0]], [[0,1], [1,1]]]), calculate_RMSE() = 515.36847 <= this is optimal solution. 
    # tf.constant([[[1, 0],[0.0]],[[1, 0],[1, 1]]]), calculate_RMSE() = 655.2762
    # tf.constant([[[0, 1],[0.0]],[[1, 0],[1, 1]]]), calculate_RMSE() = 674.356

    #indices = tf.constant([[[0,0], [1,0]], [[0,1], [1,1]]])
    indices = index_filter_tensor
    filter_tensor = tf.scatter_nd(indices, filter_tensor, filter_tensor.shape)
    print(filter_tensor)

    print(calculate_RMSE())


