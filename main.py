from aicsimageio import AICSImage
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=np.inf)
from scipy import ndimage
from skimage.measure import regionprops
from skimage.draw import polygon_perimeter
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import color
from skimage import img_as_float
from skimage.morphology import reconstruction
def read_images(filename):
    img = AICSImage(filename)
    first_channel_data = img.get_image_data('ZYX', C=0, S=0, T=0)
    return first_channel_data
def compress(first_channel_data):
    final = first_channel_data[0]
    for i in range(1,6):
        final += first_channel_data[i]
    return final
def reconstruct(img):
    image = img_as_float(img)
    # Apply gaussian filter
    image = gaussian_filter(image, 1)
    # Create a seed for dilation
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    # Create a mask for dilation
    mask = image
    # Perform dilation using the seed and mask
    dilated = reconstruction(seed, mask, method='dilation')
    return image - dilated
def count_objects(img, thresh_power=0, filter_type='median', filter_size=10, filter_power=0.15):
    threshold = np.mean(img) + thresh_power*np.std(img)
    blobs = np.where(img>threshold, 1, 0)
    if filter_type=='median':
        blobs_blur = ndimage.median_filter(blobs, size=filter_size)
    elif filter_type=='gaussian':
        blobs_blur = ndimage.gaussian_filter(blobs, filter_power)
    
    # Count the object from the image
    labels, no_objects = ndimage.label(blobs_blur)
    props = regionprops(labels)
    
    return (props, no_objects, blobs_blur)
def pipeline(filename):
    first_channel_data = read_images(filename)
    final = compress(first_channel_data)
    plt.imshow(final)
    reconstructed = reconstruct(final)
#     plt.imshow(reconstructed)
    (props, no_objects, result) = count_objects(reconstructed, thresh_power=3)
    return (props, no_objects, result)
(props, no_objects, result) = pipeline('filename2.czi')
print(no_objects)