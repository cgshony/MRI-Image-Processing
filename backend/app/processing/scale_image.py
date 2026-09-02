import numpy as np

def scale_image(image, scale_factor):
    """
    Scale the image by the given factor using nearest-neighbor interpolation.
    """
    original_height, original_width = image.shape
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    scaled_image = np.zeros((new_height, new_width))

    for i in range(new_height):  # Mapping pixels
        for j in range(new_width):
            orig_i = int(i / scale_factor)  # Corresponding row index in the original for current row
            orig_j = int(j / scale_factor)
            scaled_image[i, j] = image[orig_i, orig_j]  # Assign pixel value from the original image at pos to new

    return scaled_image
