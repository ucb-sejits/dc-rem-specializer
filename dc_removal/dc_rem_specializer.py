import omp_segmented_reduction_spec as omp_redspec
from cstructures.array import Array, specialize, wraps, gen_array_output
from ctree.util import Timer
import numpy as np
import time

#
# DC Removal
#


def dcRemoval(block_set, pfov_length, height):
    """
    Performs DC Removal with codegenerated SEJITS code. Input should be a flattened (1-D) vector.

    :param: block_set The dataset for the DC removal
    :param: pfov_length The length of the partial field of view
    :return: the result of the DC removal
    """
    shape = block_set.shape
    segmented_arr = segmented_spec(block_set, pfov_length, height)
    b = Array.array(segmented_arr / pfov_length)

    return subtract(block_set.ravel(), b, block_set.size, b.size,
                    block_set.size // height).reshape(shape, order='C')


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


#
# Mapping Specializer Definition
#


def tile_mapper(func):
    """
    Returns a function that performs a tiled mapping. This is equivalent to the following using
    numpy. (a is the bigger one) <--- TODO; check this

    >>> data1 = np.array([1] * 10)
    >>> data2 = np.tile([5, 7])
    >>> output = [func(x, y) for x, y in zip(data1, data2)]
    """
    @wraps(func)
    @specialize(output=gen_array_output)
    def fn(a, b, size_a, size_b, width, output):

        # size_b is the number of pfovs
        # modulus is the length of pfovs

        # NEW
        modulus = size_a / size_b
        for i in range(size_b):
            for j in range(modulus):
                output[i * modulus + j] = func(a[i * modulus + j], b[i])

        # OLD
        # modulus = size_a / size_b
        # for i in range(size_a):
        #     for j in range(modulus):
        #         for k in range(width):
        #             index = i * width * modulus + j * width + k
        #             output[index] = func(a[index], b[i * width + k])
        #     if i + 1 >= size_a / (width * modulus):
        #         break
    return fn


@tile_mapper
def subtract(x, y):
    return x - y


def main():

    # Smaller Dataset
    TOTAL_SIZE = 120000000
    h = 12000                # height (number of rows, or column length)
    w = 10000                 # width (number of columns, or row length)
    length = 10000

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
