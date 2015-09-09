# A specializer to perform segmented reduction that parallelizes using the OpenMP Framework.

# Importations
from __future__ import print_function, division
import numpy as np
import ctypes as ct
import time

from ctree.nodes import Project
from ctree.c.nodes import *
from ctree.c.macros import *
from ctree.cpp.nodes import *
from ctree.omp.nodes import *
from ctree.omp.macros import *
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctypes import CFUNCTYPE
from ctree.templates.nodes import StringTemplate

from ctree.transformations import PyBasicConversions
from ctree.types import get_c_type_from_numpy_dtype

from collections import namedtuple


REDUCTION_FUNC_NAME = 'reducer'


#
# Specializer Code
#


class ConcreteRemoval(ConcreteSpecializedFunction):

    """
    The actual python callable for DC Removal Specalizer.
    """

    def finalize(self, tree, entry_name, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, input_arr, stride_length, height):

        # Creating an output array; we don't want to mutate the original input data
        output_arr = np.zeros(input_arr.size // stride_length).astype(input_arr.dtype)
        self._c_function(input_arr, output_arr)
        return output_arr


class LazyRemoval(LazySpecializedFunction):

    """
    The lazy version of the DC Removal Specializer that handles code generation just in time for
    execution.
    """
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'segment_length',
                                              'data_height', 'flags'])

    def args_to_subconfig(self, args):
        input_arr = args[0]
        segment_length = args[1]
        data_height = args[2]
        return self.subconfig_type(input_arr.dtype, input_arr.ndim, input_arr.shape,
                                   input_arr.size, segment_length, data_height, [])

    def transform(self, py_ast, program_config):

        # Get the initial data
        input_data = program_config[0]

        input_length = np.prod(input_data.size)
        segment_length = np.prod(input_data.segment_length)
        data_height = np.prod(input_data.data_height)

        inner_type = get_c_type_from_numpy_dtype(input_data.dtype)()
        output_size = input_data.size // segment_length
        input_pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        output_pointer = np.ctypeslib.ndpointer(input_data.dtype, 1, (output_size, ))

        # Get the kernel function, apply_one
        apply_one = PyBasicConversions().visit(py_ast).find(FunctionDecl)

        # apply_one = PyBasicConversions().visit(py_ast.body[0])
        apply_one.return_type = inner_type
        apply_one.params[0].type = inner_type
        apply_one.params[1].type = inner_type

        # Naming our kernel method
        apply_one.name = 'apply'

        width = int(input_length / data_height)
        responsible_size = int(input_length / segment_length / width)

        reduction_template = StringTemplate(r"""
        {
            #pragma omp parallel for
            for (int k = 0; k < $width; k++) {

                #pragma omp parallel for
                for (int i = 0; i < $size; i++) {

                    float result = input_arr[i * $length * $width + k];
                    for (int j = 1; j < $length; j++) {
                        result = apply(result, input_arr[i * $length * $width + j * $width + k]);
                    }
                    output_arr[i * $width + k] = result;
                }
            }
        }
        """, {
              'width': Constant(width),
              'size': Constant(responsible_size),
              'length': Constant(segment_length)
              })

        reducer = CFile("generated", [
            CppInclude("omp.h"),
            CppInclude("stdio.h"),
            apply_one,
            FunctionDecl(None, REDUCTION_FUNC_NAME,
                params=[
                    SymbolRef("input_arr", input_pointer()),
                    SymbolRef("output_arr", output_pointer())
                ],
                defn=[
                    reduction_template
                ])
        ])

        return [reducer]

    def finalize(self, transform_result, program_config):
        tree = transform_result[0]

        # Get the argument type data
        input_data = program_config[0]
        segment_length = np.prod(input_data.segment_length)
        # data_height = input_data.data_height
        output_size = input_data.size // segment_length

        # Create the pointers for the input and output data types
        input_pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        output_pointer = np.ctypeslib.ndpointer(input_data.dtype, 1, (output_size, ))

        entry_type = CFUNCTYPE(None, input_pointer, output_pointer)

        # Instantiation of the concrete function
        fn = ConcreteRemoval()

        return fn.finalize(Project([tree]), REDUCTION_FUNC_NAME, entry_type)


#
# User Code
#


def add(x, y):
    """
    Adds two summable objects together.
    """
    return x + y


def main():

    data_size = 1000000000
    stride_length = 1000
    height = 100000
    input_array = np.array([1] * data_size)

    sum_reduction = LazyRemoval.from_function(add, "SumReducer")

    start = time.time()
    result = sum_reduction(input_array, stride_length, height)
    end = time.time()

    print ("Output Array: ", result)
    print ("Time Taken: ", end - start, " sec")

if __name__ == '__main__':
    main()
