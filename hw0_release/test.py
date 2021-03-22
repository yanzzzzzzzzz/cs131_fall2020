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

image1_path = './image1.jpg'
image1 = load(image1_path)

a = resize_image(image1, 16, 16)