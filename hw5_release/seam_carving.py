"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/16/2020
Python Version: 3.5+
"""

import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: Use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    ### YOUR CODE HERE
    dx = np.gradient(gray_image, axis=0)
    dy = np.gradient(gray_image, axis=1)
    out = np.abs(dx) + np.abs(dy)
    pass
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    In the case that energies are equal, choose the left-most path. Note that
    np.argmin returns the index of the first ocurring minimum of the specified
    axis.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    We also recommend you create a stacked matrix with left, middle, and right costs
    to make your cost and paths calculations easier.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1 (up and left), 0 (straight up), or 1 (up and right)
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1, H):
        M1 = np.insert(cost[i-1, 0:W-1], 0, 1e10, axis=0) #左側成本與路徑矩陣
        M2 = cost[i-1, :] #中間成本與路徑矩陣
        M3 = np.insert(cost[i-1, 1:W], W-1 , 1e10, axis=0) #右側成本與路徑矩陣
        M = np.r_[M1, M2, M3].reshape(3, -1) 
        cost[i] = energy[i] + np.min(M, axis=0) #更新累積最小成本
        paths[i] = np.argmin(M, axis=0) -1 #更新移動路徑
    pass
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    for i in range(H-2,-1,-1):
        seam[i] = seam[i+1] + paths[i+1,seam[i+1]]

    pass
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
             make sure that `out` has same type as `image`
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    out = image[np.arange(W) != seam[:, None]].reshape(H, W-1, C)    
    pass
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, rfunc=remove_seam):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.

    SUPER IMPORTANT: IF YOU WANT TO PREVENT CASCADING ERRORS IN THE CODE OF reduce(), USE FUNCTIONS:
        - efunc (instead of energy_function)
        - cfunc (instead of compute_cost)
        - bfunc (instead of backtrack_seam)
        - rfunc (instead of remove_seam)

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use
        bfunc: backtrack seam function to use
        rfunc: remove seam function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0: # transpose input image when different axis resize
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE

    for i in range(out.shape[1] - size):
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(H):
        idx = seam[i]
        out[i] = np.concatenate((image[i, 0:idx+1], image[i, idx:W]), axis = None).reshape(W+1, -1)  
    pass
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, dfunc=duplicate_seam):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.

    SUPER IMPORTANT: IF YOU WANT TO PREVENT CASCADING ERRORS IN THE CODE OF enlarge_naive(), USE FUNCTIONS:
        - efunc (instead of energy_function)
        - cfunc (instead of compute_cost)
        - bfunc (instead of backtrack_seam)
        - dfunc (instead of duplicate_seam)

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use
        bfunc: backtrack seam function to use
        dfunc: duplicate seam function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for i in range(size - out.shape[1]):
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = duplicate_seam(out, seam)

    pass
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, rfunc=remove_seam):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    SUPER IMPORTANT: IF YOU WANT TO PREVENT CASCADING ERRORS IN THE CODE OF find_seams(), USE FUNCTIONS:
        - efunc (instead of energy_function)
        - cfunc (instead of compute_cost)
        - bfunc (instead of backtrack_seam)
        - rfunc (instead of remove_seam)

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use
        bfunc: backtrack seam function to use
        rfunc: remove seam function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = bfunc(paths, end)

        # Remove that seam from the image
        image = rfunc(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = rfunc(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, dfunc=duplicate_seam, bfunc=backtrack_seam, rfunc=remove_seam):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    SUPER IMPORTANT: IF YOU WANT TO PREVENT CASCADING ERRORS IN THE CODE OF enlarge(), USE FUNCTIONS:
        - efunc (instead of energy_function)
        - cfunc (instead of compute_cost)
        - dfunc (instead of duplicate_seam)
        - bfunc (instead of backtrack_seam)
        - rfunc (instead of remove_seam)
        - find_seams

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use
        dfunc: duplicate seam function to use
        bfunc: backtrack seam function to use
        rfunc: remove seam function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    seams = find_seams(out, size - W)
    seams = np.expand_dims(seams, axis=2)
    for i in range(size - W):
        out = duplicate_seam(out, np.where(seams== i+1)[1])
        seams = duplicate_seam(seams, np.where(seams == i+1)[1])
    pass
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # One for-loop version
    for i in range(1,H):
        M1 = np.insert(image[i,:W-1],0,0,axis = 0)  # M1 = [0, I(i,0), I(i,1),.., I(i,W-2) ] array size:W*1
        M2 = image[i-1]                             # M2 = [I(i-1,0), I(i-1,1),..., I(i-1,W-1)] array size:W*1
        M3 = np.insert(image[i,1:W],W-1,0,axis = 0) # M3 = [I(i,1), I(i,2),.., I(i,W-1), 0] array size:W*1
        # Calculate Cu
        # Cu = [0, |I(i,0) - I(i,2)|, |I(i,1) - I(i,3)|,...,|I(i,W-3) - I(i,W-1)|, 0] 
        Cu = abs(M3 - M1) 
        # the first row and last row dont care
        Cu[0] = 0
        Cu[-1] = 0

        # Calculate CL
        # CL = [0, 
        #       |I(i,0) - I(i,2)|        + |I(i-1,1) - I(i,0)|, 
        #       |I(i,1) - I(i,3)|        + |I(i-1,2) - I(i,1)|,...,
        #       |I(i-1,W-2) - I(i,W-1)|]
        Cl = Cu + abs(M2 - M1) 
        # the first row dont care
        Cl[0] = 0
        
        # Calculate CR
        # CR = [0                        + |I(i-1,0) - I(i,1)|, 
        #       |I(i,0) - I(i,2)|        + |I(i-1,1) - I(i,2)|, 
        #       |I(i,1) - I(i,3)|        + |I(i-1,2) - I(i,3)|,...,
        #       |I(i,W-4) - I(i,W-2)|    + |I(i-1,W-2) - I(i,W-2)|,
        #       0]
        Cr = Cu + abs(M2 - M3) 
        # the last row dont care
        Cr[-1] = 0

        c1 = np.insert(cost[i-1,:W-1],0,1e10,axis = 0)
        c2 = cost[i-1]
        c3 = np.insert(cost[i-1,1:W],W-1,1e10,axis = 0) 
        
        M = np.c_[Cl+c1, Cu+c2, Cr+c3]

        cost[i] = energy[i]+np.min(M,axis = 1)
        paths[i] = np.argmin(M,axis = 1) - 1 # In y-axis find the minimum index

    # Two for-loop version
    # for i in range(1,H):
    #     M = np.zeros((W,3))
    #     for j in range(0,W):
    #         Cu =  Cr = Cl = Ml = Mu = Mr = 0
    #         if j==0:
    #             Ml = 1e10
    #             Mr = cost[i-1,j+1]
    #             Cr = np.abs(image[i-1, j] - image[i, j+1]) + Cu
    #         elif j==(W-1):
    #             Ml = cost[i-1,j-1]
    #             Mr = 1e10
    #             Cl = np.abs(image[i-1, j] - image[i, j-1]) + Cu
    #         else:
    #             Cu = np.abs(image[i, j+1] - image[i, j-1])
    #             Cr = np.abs(image[i-1, j] - image[i, j+1]) + Cu
    #             Cl = np.abs(image[i-1, j] - image[i, j-1]) + Cu
    #             Ml = cost[i-1,j-1]
    #             Mr = cost[i-1,j+1]
    #         Mu = cost[i-1,j]
    #         M[j, :] = [Cl + Ml, Cu + Mu, Cr + Mr]
    #         cost[i,j] = np.min(M[j,:]) + energy[i, j]
    #     paths[i] = np.argmin(M,axis = 1) - 1
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # Delete that line, just here for the autograder to pass setup checks
    out = reduce(image, size, 1, efunc, cfunc)
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    assert image.shape[:2] == mask.shape

    H, W, _ = image.shape
    out = np.copy(image)

    ### YOUR CODE HERE
    # process remove seam until remove all mask region 
    while len(np.where(mask[:, :]>0)[0])>0:
        energy = energy_function(out)
        # highlight mask region
        energy[np.where(mask[:, :] > 0)] *= -1000
        cost, paths = compute_cost(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)
        mask = remove_seam(mask, seam)

    # enlarge to original size
    out = enlarge(out, W)
    pass
    ### END YOUR CODE

    assert out.shape == image.shape

    return out
