import unittest
import numpy as np
from cstructures.array import Array
from dc_rem_specializer import dcRemoval as dcRemSejits
from pL_func_comparison import dcRemoval as dcRemPython


class TestDCRemovalSpecializer(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_sum_of_ones(self):

        TOTAL_SIZE = 100
        height = 10             # height: (number of rows, or column length)
        width = 10              # width: (number of columns, or row length)
        pfov_length = 5

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array([1] * TOTAL_SIZE)
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the python tests
        pyop_block_set = np.array([1] * TOTAL_SIZE)
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.float32)

        python_result = dcRemPython(pyop_block_set, height, pfov_length)
        sejits_result = dcRemSejits(sejits_block_set, pfov_length)

        self._check(python_result, sejits_result)
