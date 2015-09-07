import unittest
import numpy as np
from cstructures.array import Array, smap2, specialize, wraps, gen_array_output
import dc_rem_specializer
import pL_func_comparison
from dc_rem_specializer import dcRem as dcRemSejits
from pL_func_comparison import dcRemoval as dcRemPython


class TestDCRemovalSpecializer(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_sum_of_ones(self):

        print ("SUM OF ONES")
        TOTAL_SIZE = 100
        h = 10             # height (number of rows, or column length)
        w = 10              # width (number of columns, or row length)
        length = 5

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array([1] * TOTAL_SIZE)        # sample dataset
        sejits_block_set = sejits_block_set.reshape(h, w)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the python tests
        pyop_block_set = np.array([1] * TOTAL_SIZE)           # sample dataset
        pyop_block_set = pyop_block_set.reshape(h, w)
        pyop_block_set = pyop_block_set.astype(np.float32)

        # Setting the instance variables
        # pL_func_comparison.TOTAL_SIZE = TOTAL_SIZE
        # pL_func_comparison.h = h
        # pL_func_comparison.w = w
        # pL_func_comparison.length = length
        dc_rem_specializer.length = length

        python_result = dcRemPython(pyop_block_set, h, length)
        print ("Python Result: ", python_result)

        print ("444444")
        # block_set = Array.array([1] * TOTAL_SIZE)  # sample dataset
        # block_set = block_set.reshape(h, w)
        # block_set = block_set.astype(np.float32)
        sejits_result = dcRemSejits(sejits_block_set)
        print ("555555")

        print ("SEJITS Result: ", sejits_result)
        self._check(python_result, sejits_result)
