# Imports the print function from newer versions of python
from __future__ import print_function

# Setup

# The Random module implements pseudo-random number generators
import random 

# Numpy is the main package for scientific computing with Python. 
# This will be one of our most used libraries in this class
import numpy as np

# The Time library helps us time code runtimes
import time


# Imports all the methods in each of the files: linalg.py and imageManip.py
from linalg import *
from imageManip import *

M = np.array(range(1,13)).reshape((4, 3))
a = np.array([1, 1, 0]).reshape((1, 3))
b = np.array([-1, 2, 5]).reshape((3, 1))

print("M = \n", M)
print("The size of M is: ", M.shape)
print()
print("a = ", a)
print("The size of a is: ", a.shape)
print()
print("b = ", b)
print("The size of b is: ", b.shape)

aDotB = dot_product(a, b)
print(aDotB)

ans = complicated_matrix_function(M, a, b)
print(ans)
print()
print("The size is: ", ans.shape)