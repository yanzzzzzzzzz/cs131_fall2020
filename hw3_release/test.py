from panorama import hog_descriptor
from panorama import harris_corners
# Setup
import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
from panorama import simple_descriptor, match_descriptors, describe_keypoints
from panorama import ransac
from panorama import linear_blend
from utils import get_output_space, warp_image
from panorama import stitch_multiple_images
from PIL import Image
# Set seed to compare output against solution
np.random.seed(131)

# Load images to be stitched
ec2_img1 = imread('yosemite1.jpg', as_gray=True)
ec2_img2 = imread('yosemite2.jpg', as_gray=True)
ec2_img3 = imread('yosemite3.jpg', as_gray=True)
ec2_img4 = imread('yosemite4.jpg', as_gray=True)

imgs = [ec2_img1, ec2_img2]#, ec2_img3]#, ec2_img4]

# Stitch images together
panorama = stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5)

formatted = (panorama * 255 / np.max(panorama)).astype('uint8')
img = Image.fromarray(formatted)
img.save('save.png')

# Visualize final panorama image
plt.imshow(panorama)
plt.axis('off')
plt.title('Stiched Images')
plt.show()