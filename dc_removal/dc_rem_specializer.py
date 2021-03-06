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
    return segmented_spec(block_set, pfov_length, height, num_frames)


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
