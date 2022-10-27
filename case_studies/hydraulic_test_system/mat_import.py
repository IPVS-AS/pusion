import numpy as np
import scipy.io as sio

def load_two_parts_of_mat(filename, first_part, second_part):

    data = sio.loadmat(filename)
    first_result = np.array(data[first_part])
    second_result = np.array(data[second_part])

    return first_result, second_result




