######################################################
#  Filename: pL_func_comparison.py                   #
#  Description: This file contains the benchark DC   #
#               Removal algorithm, using the PyOP    #
#               library.                             #
######################################################

from pyop import matvectorized
from numpy import tile
from ctree.util import Timer
import numpy as np


def dcRemoval(block_set, height, length):

    @matvectorized((height, -1), order='F')
    def dcRem(block_set):

        # Partial field of views, one per row
        pfovs = block_set.reshape((-1, length))

        # Sum across rows to get the average of each pfov.
        dc_values = pfovs.sum(1) / length

        # Tiling to apply DC removal to each point in each pfov.
        # Transpose due to tile treating 1D as a row vector.
        dc_values_rep = tile(dc_values, (length, 1)).T

        # Reshape into the blocks_set image. Turn to column format
        return (pfovs - dc_values_rep).reshape((height, -1))

    return dcRem(block_set)


def main():

    # Smaller dataset
    # TOTAL_SIZE = 12000000
    # h = 12000                # height (number of rows, or column length)
    # w = 1000                 # width (number of columns, or row length)
    # length = 12

    # Larger dataset
    TOTAL_SIZE = 500000000
    h = 500000                 # height (number of rows, or column length)
    w = 1000                   # width (number of columns, or row length)
    length = 50000

    block_set = np.array(list(range(TOTAL_SIZE)))  # sample dataset
    block_set = block_set.reshape(h, w)
    block_set = block_set.astype(np.float32)

    with Timer() as t1:
        result = dcRemoval(block_set, h, length)
    time_total = t1.interval

    print "PYTHON dcRemoval Time: ", time_total, " seconds"
    print "RESULT: ", result

if __name__ == '__main__':
    main()
