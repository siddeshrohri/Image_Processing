# PILLOW (PIL) for image processing and manipulation
from PIL import Image
import numpy as np
img = Image.open("Images/cells.jpg")
print(type(img))

# img.show()
print(img.format)

# Convert to numpy 
img1 = np.asarray(img)
print(type(img1))

######################

# Matplotlib pyplot - nparray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("Images/cells.jpg")
print(type(img))
print(img.shape)
plt.imshow(img)
plt.colorbar()

#######################
# Scikit-image - Image processing, image segmentation, geometric transofmration, analysis, etc.
from skimage import io, img_as_float, img_as_ubyte

image = io.imread("Images/cells.jpg")
print(type(image))
plt.imshow(image)
print(image)
image_float = img_as_float(image)
print(image_float)
image_ubyte = img_as_ubyte(image)
print(image_ubyte)


#######################
# OpenCV - library of programming functions (Computer Vision)
import cv2
import matplotlib.pyplot as plt
grey_img = cv2.imread("Images/cells.jpg", 0)
color_img = cv2.imread("Images/cells.jpg", 1)
plt.imshow(color_img)
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
cv2.imshow("Grey image", grey_img)
cv2.imshow("Color Image", color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


######################
# Propritary files to open like czifiles - 5D images example:- (1, 513, 78, 3) 
import czifile
img = czifile.imread("")
print(img.shape)

######################
# OME-TIFF - 5D image and XML data embedded (1, 1, 3, 513, 78)
from apeer_ometiff_library import io
(pic2, omexml) = io.readometiff("")
print(pic2.shape)

#####################
# Read all images simulktanoesouly 
import cv2
import glob

path = "Images/*"
for file in glob.glob(path):
    print(file)
    a = cv2.imread(file)
    print(a)
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    cv2.imshow("Color Image", c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()