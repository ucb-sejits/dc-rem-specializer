import omp_segmented_reduction_spec as omp_redspec
from cstructures.array import Array, specialize, wraps, gen_array_output
from ctree.util import Timer
import numpy as np
import time

#
# DC Removal
#


def dcRemoval(block_set, pfov_length, height, num_frames):
    """
    Performs DC Removal with codegenerated SEJITS code. Input should be a flattened (1-D) vector.

    :param: block_set The dataset for the DC removal
    :param: pfov_length The length of the partial field of view
    :return: the result of the DC removal
    """
    shape = block_set.shape
    # print "SEJITS Real Block Shape", shape

    block_set = block_set.reshape((num_frames, -1))
    # print [frame.shape for frame in block_set]

    # Attempting to simulate matvectorized...
    block_set = np.array([frame.reshape((height, -1), order='F').flatten() for frame in block_set])

    # print "FIRST : ", block_set[0][0:5]
    # print "SECOND: ", block_set[1][0:5]

    block_set = np.expand_dims(block_set.flatten(), axis=1)
    # print "---> Shape:", block_set.shape
    # print [frame.shape for frame in block_set]

    # block_set = np.array([frame.reshape((-1, pfov_length)) for frame in block_set])

    # block_set = np.array([frame.reshape((-1, 1), order='F') for frame in block_set])
    # print [frame.shape for frame in block_set]

    # block_set = np.vstack([frame.reshape((frame.size, 1), order = 'F') for frame in block_set])
    # block_set = block_set.reshape((-1, 1), order = 'C')

    # block_set = block_set.reshape((num_frames, height, -1), order = 'F')
    # block_set = block_set.reshape((block_set.size, 1), order = 'C')

    # new_shape = temp_block.shape
    # print "SEJITS Temp Block Shape: ", temp_block.shape
    segmented_arr = segmented_spec(block_set, pfov_length, height, num_frames)

    segmented_arr = segmented_arr.reshape((num_frames, height, -1), order='C') # <------ ?????
    segmented_arr = np.array([frame.ravel(order='F') for frame in segmented_arr])
    # segmented_arr = np.array([frame.reshape((192, 52)) for frame in segmented_arr])
    # segmented_arr = segmented_arr.ravel(order='F')

    # print "SEJITS Final Shape", new_shape
    return segmented_arr.reshape((-1, 1), order='C')

#
# Segmented Reduction Specializer Invocation
#


def add(x, y):
    """
    Add two numbers (or any add-able object) together and returns the sum.

    :param: x The first object to sum
    :param: y The second object to sum
    :return: The sum of x and y
    """
    return x + y

# Generate subclass with the add() function we just defined
segmented_spec = omp_redspec.LazyRemoval.from_function(add, "SegmentedSummationSpec")

def main():

    # Smaller Dataset
    h = 1200                # height (number of rows, or column length)
    w = 1000                # width (number of columns, or row length)
    TOTAL_SIZE = h * w
    length = 100

    # Larger Dataset
    # TOTAL_SIZE = 500000000
    # h = 500000             # height (number of rows, or column length)
    # w = 1000               # width (number of columns, or row length)
    # length = 5000

    block_set = Array.array(list(range(TOTAL_SIZE)))  # sample dataset
    block_set = block_set.reshape(h, w)
    block_set = block_set.astype(np.float32)

    start_time = time.time()
    result = np.array(dcRemoval(block_set.flatten(), length, h).reshape(h, w))
    time_total = time.time() - start_time

    print "SEJITS dcRemoval Time: ", time_total, " seconds"
    print "RESULT: ", result
    return result

if __name__ == '__main__':
    main()
