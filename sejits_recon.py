#!/usr/bin/env python3

from numpy import (
        reshape, tile, hsplit, hstack,
        zeros, array, swapaxes)

from math import sqrt

from scipy.sparse.linalg import eigsh

from scipy.ndimage.interpolation import zoom

from scipy.io import loadmat, savemat

from pyop import LinearOperator, toScipyLinearOperator, matvectorized
from pyop.block import vstack, blockDiag

from pyop.operators import convolve, eye

from collections import namedtuple

from itertools import tee, count

import time

import six

from dc_removal.dc_rem_specializer import dcRemoval
from cstructures.array import Array

def fista(A, b, pL, initial = None,
        residual_diff = 0.0001, max_iter = 1000,
        monotone = False, F = None, logger=lambda x, v: x):
    r''' Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    FISTA is an algorithm for solving linear inverse problems. It stems
    from the iterative shrinkage-thresholding algorithms (ISTA) class of
    methods, which normally converge quite slowly. This technique keeps the
    simplicity of ISTA while reaching a faster O(1/k^2) convergence rate in
    both theory and practice.

    Parameters
    ----------
    A : array like (numpy.ndarray, numpy.matrix, LinearOperator)
        2D array representing the linear operator. Note: does not have to
        be square

    b : array like (numpy.ndarray, numpy.matrix, LinearOperator)
        Vector of the received data (the result from the forward
        transformation).

    pL : function R^n -> R^n
        See the notes section.

    residual_diff : float, optional
        Specifies when to stop looping, (when the change in the residuals is
        less than this value).

    max_iter : int, optional
        The maximum number of iterations in the algorithm. If the residual
        difference is not yet below the specified threshold, then the
        solution at the last iteration is returned.

    monotone : bool, optional
        Whether to use the monotone version of the gradient descent or not
        (the residuals are monotonic).

    F : function R^n -> R
        The objective function to be minimized. The monotone flag requires
        this parameter, which is by default None. Therefore, in order to
        use the monotone version, this parameter also must be explicitly
        specified.

        A common function is :math:`F = x \mapsto \|A \cdot x - b\|^2`.


    Returns
    -------
    x* : numpy.ndarray
        The optimal solution to within a certain accuracy.

    residual : numpy.ndarray of float
        The residuals accrued during the looping. This also contains a
        record of how many iterations were performed (its length).

    Notes
    -----
    The FISTA algorithm solves the following minimization problem.

    ..math:: \min_x \{ F \equiv f(x) + g(x) \}

    where :math:`f` is a smooth convex function of type :math:`C^{1,1}`
    (i.e. continuously differentiable with Lipschitz continuous gradient
    :math:`L(f)`), and :math:`g` is convex but not necessarily smooth. A
    general example of this type of problem is

    ..math:: \min_x \| Ax-b \|^2 + \lambda \|x\|_1

    The function :math:`p_L` is a function that turns the data in the
    co-domain back into the original domain. This equation must follow the
    description below.

    .. math::

        p_L(y) = \mathrm{argmin}_x g(x) + \frac{L}{2} \left\|x -
            \left(y - \frac{1}{L}\nabla f(y) \right) \right\|^2

    .. [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems," SIAM J. Imaging Sci., vol.
       2, no. 1, pp. 183-202, Jan. 2009. http://dx.doi.org/10.1137/080716542

    .. [2] A. Beck and M. Teboulle, "Gradient-Based Algorithms with
       Applications to Signal Recovery".
       http://iew3.technion.ac.il/~becka/papers/gradient_chapter.pdf
    '''
    from numpy.linalg import norm
    from scipy.sparse.linalg import eigsh
    from numpy import zeros, squeeze, sqrt

    if monotone and F is None:
        TypeError('To use monotone FISTA an F must be provided.')

    b = squeeze(b)

    ## Initial guess is empty
    if initial is None:
        y = zeros(A.shape[1])
    else:
        y = initial

    x1 = y

    t1 = 1

    ## Initialize residual using empty guess

    # print "A's shape: ", A.shape
    # print "b's shape: ", b.shape
    # print "x1's shape: ", x1.shape
    residual = [ norm(A.dot(x1) - b) / norm(b) ]

    i = 0
    while True:

        ## Calculate the next optimal solution.
        z = pL(y)
        i += 1

        ## Change ratio of how much of the last two solutions to mix. The
        ## ratio decreases over time (the solution calculated from pL is
        ## more accurate).
        t2 = 0.5*(1+sqrt(1+4*t1**2))

        if monotone:
            x2 = min((z, x1), key = F)
        else:
            x2 = z

        ## Used to reduce the number of calculations from the algorithms
        ## given in the gradient chapter. Use 'is' for constant time lookup.
        if x2 is x1:
            mix = (t1/t2) * (z - x2)
        else: ## x2 is z
            mix = ((t1 - 1)/t2) * (x2 - x1)

        ## Create next mixed iteration vector
        y = x2 + mix

        ## Calculate the norm residuals of this new result
        residual.append(norm(A.dot(x2) - b) / norm(b))
        logger("iteration {}, residual: {}".format(i, residual[-1]), 2)

        ## Break here since we don't need a new y if the change in residual
        ## is below the desired cutoff.
        if (residual[-2] - residual[-1]) < residual_diff or i >= max_iter:
            break

        ## Set up for next iteration.
        x1 = x2
        t1 = t2
        print "Completed FISTA iteration"

    ## z is returned because it is the last calculated value that adheres
    ## to any constraints inside pL (positivity).
    return z, residual

#######################################################################
#                      Utilities/Data Structures                      #
#######################################################################
__ImageShape = namedtuple('ImageShape', 'height width')
__Blocks = namedtuple('Blocks', 'len height num shift')

PfovImage = namedtuple('PfovImage', 'planes width shift num shape scan_dir numPixGrid')


def __toImageShapeAndBlocks(image_shape, blocks):

    h, w = image_shape
    length, shift = blocks

    image_shape = __ImageShape(*image_shape)
    blocks = __Blocks(length, h, int((w - length) / shift) + 1, shift)

    if any(x <= 0 for x in image_shape):
        raise ValueError("Image shape contains a non-positive. {}".
                         format(image_shape))

    if any(x <= 0 for x in blocks):
        raise ValueError("blocks contains a non-positive. {}".
                         format(blocks))

    return image_shape, blocks


def __toImageShapeAndBlocksSejits(image_shape, blocks):

    h, w, num_frames = image_shape
    length, shift = blocks

    image_shape = __ImageShape(*(image_shape[:2]))
    blocks = __Blocks(length, h, int((w - length) / shift) + 1, shift)

    if any(x <= 0 for x in image_shape):
        raise ValueError("Image shape contains a non-positive. {}".
                         format(image_shape))

    if any(x <= 0 for x in blocks):
        raise ValueError("blocks contains a non-positive. {}".
                         format(blocks))

    return image_shape, blocks


def __pairwise(x):
    ''' s --> (s0, s1), (s1, s2), (s2, s3), ...  '''
    a, b = tee(x)
    next(b, None)
    return six.moves.zip(a, b)


#######################################################################
#                              Operators                              #
#######################################################################

def splittingOperator(image_shape, blocks):
    '''
    Splitting cuts apart the image into the different partial fields of view
    (pFOV).

    Parameters
    ----------
    image_shape : pair (height, width)
        An pair containing the height and width of the image to perform a
        transformation on.

    blocks : pair (len, shift)
        A pair containing the length of a block (the length of a partial
        field of view) and how much each block is shifted relative to its
        neighbors.

    Returns
    -------
    S : pyop.LinearOperator
        A LinearOperator version of the splitting operator.

    See Also
    --------
    paddingOperator   : Pads the original image before splitting.
    dcRemovalOperator : Splits a row flattened vector image into blocks.
    '''

    #pylint: disable=W0612,W0613
    image_shape, blocks = __toImageShapeAndBlocks(image_shape, blocks)

    h, w = image_shape
    length, _, num, shift = blocks

    op_shape = (length * h * num, h * w)

    def stride(x, shift, length):
        i = 0

        ## Run until we can't take any more lengths out
        while i + length <= len(x):
            yield x[i: i + length]
            i += shift


    @matvectorized((h, w), order = 'F')
    def splitting(img):
        ## The img.T is to iterate over the columns, while the z.T is to
        ## turn the chunks returned by stride back into the untransformed
        ## view. Flatten along the columns.
        return hstack(z.T for z in stride(img.T, shift, length))


    @matvectorized((h, -1), order='F')
    def splittingAdjoint(block_set):
        block_list = hsplit(block_set, num)

        combined = zeros((h, w))
        for step, b in six.moves.zip(count(step = shift), block_list):
            ## Add in each block in the right spot
            combined[:, step : step + length] += b

        return combined.flatten(1)


    return LinearOperator(op_shape, splitting, splittingAdjoint)


def dcRemovalOperatorSejits(image_shape, blocks):
    ''' Removes the DC components of an MPI signal.

    Filtering by the MPI system removes the DC component for each partial
    field of view (pFOV). dcRemovalOperator subtracts the mean value from
    each pFOV to remove the "false" DC in the receive signal. The operator
    is symmetric and self-adjoint.

    Parameters
    ----------
    image_shape : pair (height, width)
        An pair containing the height and width of the image to perform a
        transformation on.

    blocks : pair (len, shift)
        A pair containing the length of a block (the length of a partial
        field of view) and how much each block is shifted relative to its
        neighbors.

    Returns
    -------
    D : pyop.LinearOperator
        A LinearOperator version of the DC removal operator.

    See Also
    --------
    paddingOperator   : Pads the original image before splitting.
    splittingOperator : Splits a row flattened vector image into blocks.
    '''

    num_frames = image_shape[2]
    image_shape, blocks = __toImageShapeAndBlocksSejits(image_shape, blocks)

    h, _ = image_shape
    length, _, num, _ = blocks

    op_size = length * h * num * num_frames
    op_shape = (op_size, op_size)  # This needs to change, maybe?

    ## @Mihir: this is where your code was inserted
    sejits_dcrem = lambda block_set: dcRemoval(Array.array(block_set), length, h, num_frames)

    # @matvectorized((h, -1), order = 'F')  # TODO: this could be a problem...
    def dcRem(block_set):

        # Matvectorize each layer

        # block_set = block_set.reshape((block_set.size, 1))
        # y_coord = block_set.size // num_frames // length
        # frame_size = block_set.size // num_frames
        # print "Started dcRem SEJITS with block shape: ({0}, {1}, {2})".format(num_frames, y_coord, length)
        # print "SEJITS Frame 1 Starters: ", block_set[:5].flatten()
        # print "SEJITS Frame 2 Starters: ", block_set[frame_size:5+frame_size].flatten()
        ## @Mihir: this is where your code was inserted
        return sejits_dcrem(block_set)

    return LinearOperator(op_shape, dcRem, dcRem)

def dcRemovalOperatorPyOp(image_shape, blocks):
    ''' Removes the DC components of an MPI signal.

    Filtering by the MPI system removes the DC component for each partial
    field of view (pFOV). dcRemovalOperator subtracts the mean value from
    each pFOV to remove the "false" DC in the receive signal. The operator
    is symmetric and self-adjoint.

    Parameters
    ----------
    image_shape : pair (height, width)
        An pair containing the height and width of the image to perform a
        transformation on.

    blocks : pair (len, shift)
        A pair containing the length of a block (the length of a partial
        field of view) and how much each block is shifted relative to its
        neighbors.

    Returns
    -------
    D : pyop.LinearOperator
        A LinearOperator version of the DC removal operator.

    See Also
    --------
    paddingOperator   : Pads the original image before splitting.
    splittingOperator : Splits a row flattened vector image into blocks.
    '''

    image_shape, blocks = __toImageShapeAndBlocks(image_shape, blocks)

    h, _ = image_shape
    length, _, num, _ = blocks

    op_size = length * h * num
    op_shape = (op_size, op_size)

    @matvectorized((h, -1), order = 'F')
    def dcRem(block_set):
        # print ("Hello: ", h)

        # print "Block Set Sum Pyop:", sum(block_set.flatten())
        # print "----- : ", block_set.flatten()[0:5]

        # print "Pyop input shape", block_set.shape
        # print "PYOP:", block_set.flatten()[:5]

        # print "dcRem Shape", block_set.shape
        ## Partial field of views, one per row
        pfovs = block_set.reshape((-1, length))
        # print "Started dcRem PyOP with block shape:", pfovs.shape

        # print "Pyop Num Pfovs", len(pfovs)
        # print "Pyop Pfov Length", length
        # print "Pyop PFOVS SHAPE:", pfovs.shape



        ## Sum across rows to get the average of each pfov.
        dc_values = pfovs.sum(1)
        # print "Length Pyop:", len(dc_values)
        # print "PFOV Length", length
        # print "PFOV Height", h

        # print "PYOP:", sum(dc_values)

        ## The important one
        # print "Pyop DC Values:", dc_values[:5]

        dc_values = dc_values / length

        ## Tiling to apply DC removal to each point in each pfov.
        ## Transpose due to tile treating 1D as a row vector.
        dc_values_rep = tile(dc_values, (length, 1)).T

        ## reshape into the blocks_set image. Turn to column format
        result = (pfovs - dc_values_rep).reshape((h, -1))
        # print "PyOP Result Shape: ", result.shape
        return result



    ## @Mihir: this is where your code was inserted
    # sejits_dcrem = lambda block_set: dcRemoval(Array.array(block_set), length, h)
    # return LinearOperator(op_shape, sejits_dcrem, sejits_dcrem)
    return LinearOperator(op_shape, dcRem, dcRem)

def dc_recon(pfovimage, tikhonov = 0.0, smooth = 0.0,
             residual_diff = 0.001, max_iter = 100,
             logger = None):
    ''' Optimized 3D DC Reconstruction

    This performs the

    Parameters
    ----------
    pfovimage : PfovImage
        A PfovImage structure. The images in the structure need to have
        vertical blocks.

        Example (where the `O`s represent overlap)

        ```
        +----+--+----+--+----+--+----+--+----+
        |    |OO|    |OO|    |OO|    |OO|    |
        |    |OO|    |OO|    |OO|    |OO|    |
        +----+--+----+--+----+--+----+--+----+
        ```

    tikhonov : float
        The Tikhonov regularization parameter.
    smooth : float
        The smoothing regularization parameter.
    residual_diff : float
        The difference between iterations of the solver at which to stop, or
        if the maximum number of iterations is hit.
    max_iter : int
        The maximum number of iterations to run the solver. It may stop
        before this number if the residual difference criteria is met.
    logger : function (string, int)
        A function taking in a message and a verbosity priority.

    Returns
    -------
    x* : numpy.ndarray
        The optimal solution to within a certain accuracy.

    residual : numpy.ndarray of float
        The residuals accrued during the looping. This also contains a
        record of how many iterations were performed (its length).
    '''

    if logger is None:
        logger = lambda x, v: None

    ###########################
    #  Load in the pfov file  #
    ###########################

    (planes, width, shift, num, shape, scan_dir, numPixGrid) = pfovimage
    frames = shape[2]


    ##########################
    #  Create the Operators  #
    ##########################

    ## Create operators and artificial image.
    ## Do linop stuff to create P, S, D
    S = splittingOperator(shape[:2], (width, shift))
    Dpyop = dcRemovalOperatorPyOp(shape[:2], (width, shift))
    Dsejits = dcRemovalOperatorSejits(shape, (width, shift))

    Apyop = Dpyop*S
    Asejits = Dsejits*blockDiag([S] * frames)

    ## TODO: Add the different smoothing parameters instead of the 1 and -1
    ## values in this array. The names of the variables should be something
    ## like bi, bj, bk since the physical x, y, and z directions change
    ## based on whether this is a y or z scan.
    kernel = array(
            [
              [ [0 ,  0, 0],
                [0 , -1, 0],
                [0 ,  0, 0] ] ,

              [ [0 , -1, 0],
                [-1,  0, 1],
                [0 ,  1, 0] ] ,

              [ [0 ,  0, 0],
                [0 ,  1, 0],
                [0 ,  0, 0] ]
            ]
            )

    S_hat = blockDiag([S] * frames)

    vec_size = shape[0]*shape[1]*shape[2]
    A_hat_pyop = vstack(    # not actually A_hat blockDiag is A_hat
        [ blockDiag([Apyop] * frames),
          sqrt(tikhonov)*eye((vec_size, vec_size)),
          sqrt(smooth)*convolve(kernel = kernel, shape = shape, order = 'F')
        ])

    A_hat_sejits = vstack(
        [ Asejits,
          sqrt(tikhonov)*eye((vec_size, vec_size)),
          sqrt(smooth)*convolve(kernel = kernel, shape = shape, order = 'F')
        ])


    # NOTE: If you're going to get rid of convultion, make sure to make use 1*vec_size instead of 2*vec_size
    planes_vec = hstack([f.flatten(1) for f in planes])
    b = hstack( [planes_vec, zeros((2*vec_size, ))] )


    ## Largest possible safe step size. If the alpha is any larger then the
    ## algorithm explodes (the norm increases per iteration).
    Bpyop = toScipyLinearOperator(A_hat_pyop.T*A_hat_pyop)
    Bsejits = toScipyLinearOperator(A_hat_sejits.T*A_hat_sejits)

    descent_step_pyop = 1/(eigsh(Bpyop, 1, tol = 4)[0][0])
    descent_step_sejits = descent_step_pyop # 1/(eigsh(Bsejits, 1, tol = 4)[0][0])

    print "Descent Step Pyop:", descent_step_pyop
    print "Descent Step SEJITS:", descent_step_sejits

    print "A Hat Pyop:", A_hat_pyop
    print "A Hat SEJITS:", A_hat_sejits

    ## Projection onto the positive orthant
    def pL_pyop(y):
        z = y - descent_step_pyop*(A_hat_pyop.T*(A_hat_pyop*y - b))
        z[z<0] = 0
        return z

    def pL_sejits(y):
        z = y - descent_step_sejits*(A_hat_sejits.T*(A_hat_sejits*y - b))
        z[z<0] = 0
        return z

    logger('Starting PyOp FISTA iteration', 1)
    start_time = time.time()
    x, res_pyop = fista(A_hat_pyop, b, pL_pyop, initial = S_hat.T*planes_vec,
            residual_diff = residual_diff,
            max_iter = max_iter,
            logger = logger)
    print "Total PyOp FISTA Time:", time.time() - start_time

    image_pyop = reshape(x, shape, order='F')
    image_pyop = zoom(image_pyop, (1, float(numPixGrid)/shape[1], 1))

    ## The z data comes in as stacks of xz or yz planes, so the array needs
    ## to be flipped to match the expected axes.
    if scan_dir is "z":
        image_pyop = swapaxes(image_pyop, 0, 1).T

    logger("Residual FISTA: {}".format(res_pyop[-1]), 1)
    logger("Number of iterations: {}".format(len(res_pyop) - 1), 1)



    logger('Starting SEJITS FISTA iteration', 1)
    start_time = time.time()
    x, res_sejits = fista(A_hat_sejits, b, pL_sejits, initial = S_hat.T*planes_vec,
            residual_diff = residual_diff,
            max_iter = max_iter,
            logger = logger)
    print "Total SEJITS FISTA Time:", time.time() - start_time

    image_sejits = reshape(x, shape, order='F')
    image_sejits = zoom(image_sejits, (1, float(numPixGrid)/shape[1], 1))

    ## The z data comes in as stacks of xz or yz planes, so the array needs
    ## to be flipped to match the expected axes.
    if scan_dir is "z":
        image_sejits = swapaxes(image_sejits, 0, 1).T

    logger("Residual FISTA: {}".format(res_sejits[-1]), 1)
    logger("Number of iterations: {}".format(len(res_sejits) - 1), 1)

    return (image_pyop, res_pyop), (image_sejits, res_sejits)


if __name__ == '__main__':
    import argparse


    #######################
    #  Program Arguments  #
    #######################

    parser = argparse.ArgumentParser(description="Optimized DC Recon")
    parser.add_argument("file", metavar="FILE", type=str,
        help="The partial field of view file to process")

    parser.add_argument("-a", "--tikhonov",
            help=
                "Specify the tikhonov "
                "regularization value. A higher value makes the "
                "reconstruction try harder to make the DC values close "
                "to zero.",
            type=float,
            default=0.0)
    parser.add_argument("-b", "--smooth",
            help=
                "Specify the smoothing "
                "regularization value. A higher value makes the "
                "reconstruction try harder to make the image smooth.",
            type=float,
            default=0.0)

    parser.add_argument("-r", "--residual_diff",
            help= "Specify the stopping condition for reconstruction.",
            type=float,
            default=0.001)
    parser.add_argument("-i", "--max_iter",
            help= "Specify the maximum number of iterations",
            type=int,
            default=100)

    parser.add_argument("-s", "--save",
            help= "Change where the output is saved",
            type=str,
            default="dc_optim_recon.mat")
    parser.add_argument("--stdout",
            help= "Output file to stdout",
            action="store_true")

    parser.add_argument("-v", "--verbose",
            help= "Print out messages during reconstruction",
            action="count")

    args = parser.parse_args()


    ##############
    #  Defaults  #
    ##############

    def verbose_logger(x, v):
        if v <= args.verbose:
            print(x)
        return None

    if args.verbose:
        logger = verbose_logger
    else:
        logger = lambda x, v: None

    if args.stdout:
        import sys
        save = sys.stdout
    else:
        save = args.save

    logger("Starting 3D Optimization DC Reconstruction", 1)

    ## Now lets do some actual processing!
    mat = loadmat(args.file, squeeze_me = True)

    pfovimage = PfovImage(
        planes     = mat["pfov_images"],
        width      = int(mat["pfov_width"]),
        shift      = int(mat["pfov_width"] - mat["pfov_overlap"]),
        num        = int(mat["pfov_num"]),
        shape      = (int(mat["image_shape"][0]),
                      int(mat["image_shape"][1]),
                      int(mat["image_shape"][2])),
        scan_dir   = mat["scan_dir"],
        numPixGrid = mat["numPixGrid"]
        )

    (image_pyop, res_pyop), (image_sejits, res_sejits) = dc_recon(pfovimage,
            args.tikhonov, args.smooth,
            args.residual_diff, args.max_iter,
            logger)

    savemat(save.strip(".mat") + "_sejits.mat", {'image': image_sejits, 'residuals': res_sejits})
    savemat(save.strip(".mat") + "_pyop.mat",   {'image': image_pyop, 'residuals': res_pyop})
