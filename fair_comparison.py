
from dc_removal.dc_rem_specializer import dcRemoval as dcRemovalSejits
from pyop import matvectorized
from numpy import tile
import numpy as np
import time

# Setting up the input data
num_frames = 20
pfov_length = 1000

data_height = 100
data_width = 1000

total_size = num_frames * data_height * data_width
input_data = np.array(list(range(total_size))).reshape((num_frames, -1))

print "Input size: ", total_size


# Setting up PyOP
def dcRemovalPyop(height, length):
    """
    :block_set: The block_set
    :height: The height of the data
    :length: The length of each of the pfov
    """
    @matvectorized((height, -1), order='F')
    def dcRem(block_set):
        pfovs = block_set.reshape((-1, length))
        dc_values = pfovs.sum(1) / length
        dc_values_rep = tile(dc_values, (length, 1)).T
        return (pfovs - dc_values_rep).reshape((height, -1))

    return dcRem

dcRemPyop = dcRemovalPyop(data_height, pfov_length)

# PyOP
pyop_start_time = time.time()
pyop_result = np.array([dcRemPyop(frame) for frame in input_data])
pyop_total_time = time.time() - pyop_start_time
print "PyOP Time  : ", pyop_total_time


# SEJITS
sejits_start_time = time.time()
sejits_result = dcRemovalSejits(input_data, pfov_length, data_height, num_frames)
sejits_total_time = time.time() - sejits_start_time
print "SEJITS Time: ", sejits_total_time

print "Identical Results?: ", all(pyop_result.flatten() == sejits_result.flatten())
