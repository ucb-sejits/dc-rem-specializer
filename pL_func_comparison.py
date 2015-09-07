######################################################
#  Filename: pL_func_comparison.py                   #
#  Description: This file contains the benchark DC   #
#               Removal algorithm, using the PyOP    #
#               library.                             #
######################################################

from pyop import matvectorized
from numpy import tile


def dcRemoval(block_set, height, length):

    @matvectorized((height, -1), order='F')  # <- what does that do?
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
