# Image processing using scikit learning
from skimage import io
from matplotlib import pyplot as plt

# tranformations
img = io.imread("Images/cells.jpg", as_gray = True)
from skimage.transform import rescale, resize, downscale_local_mean
rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)
resized_img = resize(img, (200,200))
downscaled_img = downscale_local_mean(img, (4,3))
plt.imshow(rescaled_img)
plt.imshow(resized_img)
plt.imshow(downscaled_img)

# edge detection and convolution
from skimage.filters import roberts, sobel, scharr, prewitt
edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)
plt.imshow(edge_prewitt, cmap="gray")
# Canny edge detector - In features module of skimage, its binary output
from skimage.feature import canny
edge_canny = canny(img, sigma = 3)
plt.imshow(edge_canny)