from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float

# Read the image
my_image = io.imread("Images/cells.jpg")
print(my_image.min(), my_image.max())
plt.imshow(my_image)

# Conversion of image to float type
my_float_img = img_as_float(my_image)
print(my_float_img.min(), my_float_img.max())

# Random image generation
random_image = np.random.random([500, 500])
plt.imshow(random_image)
print(random_image.min(), random_image.max())

# Changing values of the pixels
dark_image = my_image * 0.5
print(dark_image)
plt.imshow(dark_image)

# Conversion of values of the images
my_image[10:200, 10:200, :] = [255, 255, 0]
plt.imshow(my_image)
