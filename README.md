DC Removal Specializer
======================
*A SEJITS Specializer for DC Removal*


What is DC Removal?
-------------------

DC Removal is the process of making a list of values zero-mean. For example, if
our input dataset was an array like `[1, 2, 3, 4, 5]`, which has an average of
3, after performing DC Removal, our output array would look like `[-2, -1, 0, 1, 2]`,
where the mean, 3 is subtracted out from each element of the original dataset. Notice
how the output dataset has a mean of 0, or in other words, the dataset is zero-mean.

Installation
------------

**Prerequisites**

In order to use this specializer, certain prerequisite libraries must be installed. Here
we use `pip` for our installation process. In your shell, outside your project directory,
run the following commands:

    $ pip install ctree                                          # installs the ctree library
    $ git clone https://github.com/ucb-sejits/cstructures.git    # clones the cstructures library
    $ cd cstructures
    $ pip install -e .                                           # installs the development version of cstructures
    $ cd ..
    $ pip install scipy                                          # install scipy
    $ pip install nose                                           # install nose
    $ pip install mako                                           # install mako
    
**Library Installation**

Once we have installed the prerequiste libraries, we need to install the development version
of this specializer. You can do this with the following commands from the same directory as above.

    $ git clone https://github.com/ucb-sejits/dc-rem-specializer.git  # clones the DC Removal Specializer
    $ cd dc-rem-specializer
    $ pip install -e .                                                # installs the DC Removal Specializer
    $ cd ..

Congratulations, installation is complete!

Running Tests
-------------

To check for successful installation, run the following:

    $ python -m nose                                                  # running nose tests
    
All tests should pass.
    
Usage
-----

To use the specializer we first do the following importations.

    from dc_removal.dc_rem_specializer import dcRemoval
    from cstructures.array import Array
    
Next, we construct our input dataset using `cstructures.array` (of type `Array`), which is a subclass of `numpy.array`. Let's call this input data set `input_data`. Now, we can just make the following call:

    output_data = dcRemoval(input_data, segmentation_length, stride_height)








