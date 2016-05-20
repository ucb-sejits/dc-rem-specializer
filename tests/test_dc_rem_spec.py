import unittest
import numpy as np
from cstructures.array import Array
from dc_removal.dc_rem_specializer import dcRemoval as dcRemSejits
from dc_removal_python import dcRemoval as dcRemPython


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

        python_result = dcRemPython(pyop_block_set, height, pfov_length).astype(np.float32)
        sejits_result = np.array(
            dcRemSejits(sejits_block_set, pfov_length, height)).astype(np.float32)

        self._check(python_result, sejits_result)

    def test_range_of_ten(self):

        TOTAL_SIZE = 10
        height = 10             # height: (number of rows, or column length)
        width = 1               # width: (number of columns, or row length)
        pfov_length = 5

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array(list(range(TOTAL_SIZE)))
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the Python tests
        pyop_block_set = np.array(list(range(TOTAL_SIZE)))
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.float32)

        python_result = dcRemPython(pyop_block_set, height, pfov_length)
        sejits_result = np.array(dcRemSejits(sejits_block_set, pfov_length, height))

        self._check(python_result, sejits_result)

    def test_range_of_fifty(self):

        TOTAL_SIZE = 50
        height = 25             # height: (number of rows, or column length)
        width = 2               # width: (number of columns, or row length)
        pfov_length = 5

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array(list(range(TOTAL_SIZE)))
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the Python tests
        pyop_block_set = np.array(list(range(TOTAL_SIZE)))
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.float32)

        python_result = dcRemPython(pyop_block_set.flatten(1), height, pfov_length).astype(
            np.float32).reshape((height, width), order='F').astype(np.int32)
        sejits_result = np.array(
            dcRemSejits(sejits_block_set.flatten(), pfov_length, height)).astype(
            np.float32).reshape((height, width)).astype(np.int32)

        print sejits_result

        self._check(python_result, sejits_result)

    def test_range_of_hundred(self):

        TOTAL_SIZE = 100
        height = 20             # height: (number of rows, or column length)
        width = 5               # width: (number of columns, or row length)
        pfov_length = 4

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array(list(range(TOTAL_SIZE)))
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the Python tests
        pyop_block_set = np.array(list(range(TOTAL_SIZE)))
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.float32)

        python_result = dcRemPython(pyop_block_set.flatten(1), height, pfov_length).astype(
            np.float32).reshape((height, width), order='F')
        sejits_result = np.array(
            dcRemSejits(sejits_block_set.flatten(), pfov_length, height)).astype(
            np.float32).reshape((height, width))

        self._check(python_result, sejits_result)

    def test_range_of_two_hundred(self):

        TOTAL_SIZE = 200
        height = 20             # height: (number of rows, or column length)
        width = 10              # width: (number of columns, or row length)
        pfov_length = 10

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array(list(range(TOTAL_SIZE)))
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.float32)

        # Creating a sample dataset for the Python tests
        pyop_block_set = np.array(list(range(TOTAL_SIZE)))
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.float32)

        python_result = dcRemPython(pyop_block_set.flatten(1), height, pfov_length).astype(
            np.float32).reshape((height, width), order='F')
        sejits_result = np.array(
            dcRemSejits(sejits_block_set.flatten(), pfov_length, height)).astype(
            np.float32).reshape((height, width))

        self._check(python_result, sejits_result)

    def test_range_of_two_hundred_ints(self):

        TOTAL_SIZE = 200
        height = 20             # height: (number of rows, or column length)
        width = 10              # width: (number of columns, or row length)
        pfov_length = 10

        # Creating a sample dataset for the SEJITS tests
        sejits_block_set = Array.array(list(range(TOTAL_SIZE)))
        sejits_block_set = sejits_block_set.reshape(height, width)
        sejits_block_set = sejits_block_set.astype(np.int32)

        # Creating a sample dataset for the Python tests
        pyop_block_set = np.array(list(range(TOTAL_SIZE)))
        pyop_block_set = pyop_block_set.reshape(height, width)
        pyop_block_set = pyop_block_set.astype(np.int32)

        python_result = dcRemPython(pyop_block_set.flatten(1), height, pfov_length).astype(
            np.float32).reshape((height, width), order='F').astype(np.int32)
        sejits_result = np.array(
            dcRemSejits(sejits_block_set.flatten(), pfov_length, height)).astype(
            np.float32).reshape((height, width)).astype(np.int32)

        self._check(python_result, sejits_result)
