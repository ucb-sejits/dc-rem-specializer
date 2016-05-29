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

    def __call__(self, input_arr, stride_length, height, num_frames):

        # Creating an output array; we don't want to mutate the original input data

        # print ("__CALL__")
        output_arr = np.zeros((input_arr.size, )).astype(input_arr.dtype)
        # print ("__CALL__ 222")
        self._c_function(input_arr, output_arr)
        # print ("__CALL__ 333")
        return output_arr.reshape(input_arr.shape)


class LazyRemoval(LazySpecializedFunction):

    """
    The lazy version of the DC Removal Specializer that handles code generation just in time for
    execution.
    """
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'segment_length',
                                              'data_height', 'num_frames', 'flags'])

    def args_to_subconfig(self, args):
        input_arr = args[0]
        segment_length = args[1]
        data_height = args[2]
        num_frames = args[3]
        return self.subconfig_type(input_arr.dtype, input_arr.ndim, input_arr.shape,
                                   input_arr.size, segment_length, data_height, num_frames, [])

    def transform(self, py_ast, program_config):

        # Get the initial data
        input_data = program_config[0]

        num_2d_layers = np.prod(input_data.num_frames)
        data_height = np.prod(input_data.data_height)
        layer_length = np.prod(input_data.size // num_2d_layers)
        segment_length = np.prod(input_data.segment_length)

        inp_type = get_c_type_from_numpy_dtype(input_data.dtype)()

        input_pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        output_pointer = np.ctypeslib.ndpointer(input_data.dtype, 1, (input_data.size, 1))

        # Get the kernel function, apply_one
        apply_one = PyBasicConversions().visit(py_ast).find(FunctionDecl)

        apply_one.return_type = inp_type
        apply_one.params[0].type = inp_type
        apply_one.params[1].type = inp_type

        # Naming our kernel method
        apply_one.name = 'apply'
        num_pfovs = int(layer_length / segment_length)
        print ("num layers: ", num_2d_layers)
        print ("input size: ", input_data.size)
        print ("layer length: ", layer_length)

        # TODO: TIME TO START CPROFILING THINGS!
        reduction_template = StringTemplate(r"""
            #pragma omp parallel for // collapse(2)
            for (int level = 0; level < $num_2d_layers; level++) {
                int level_offset = level * $layer_length;

                for (int i=0; i<$num_pfovs ; i++) {
                    int raw_index = 0, index = 0, count = 0;
                    double avg = 0.0;
                    for (int j=0; j<$pfov_length; j++) {
                        int in_layer_offset = ($pfov_length * i + j) /
                            ($layer_length / $data_height);

                        raw_index = in_layer_offset + ($pfov_length * i + j) * $data_height;
                        index = raw_index % $layer_length;
                        // printf ("Index: %i, I: %i, J: %i\n", index, i, j);
                        avg += input_arr[level_offset + index];
                    }
                    avg = avg / $pfov_length;

                    for (int j=0; j<$pfov_length; j++) {
                        int in_layer_offset = ($pfov_length * i + j) /
                            ($layer_length / $data_height);

                        raw_index = in_layer_offset + ($pfov_length * i + j) * $data_height;
                        index = raw_index % $layer_length;
                        output_arr[level_offset + index] = input_arr[level_offset + index] - avg;
                    }
                }
            }
        """, {
            'num_2d_layers': Constant(num_2d_layers),
            'layer_length': Constant(layer_length),
            'num_pfovs': Constant(num_pfovs),
            'pfov_length': Constant(segment_length),
            'data_height': Constant(data_height),
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
        ], 'omp')

        return [reducer]

    def finalize(self, transform_result, program_config):
        tree = transform_result[0]

        # Get the argument type data
        input_data = program_config[0]

        # Create the pointers for the input and output data types
        input_pointer = np.ctypeslib.ndpointer(input_data.dtype, input_data.ndim, input_data.shape)
        output_pointer = np.ctypeslib.ndpointer(input_data.dtype, 1, (input_data.size, ))

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
