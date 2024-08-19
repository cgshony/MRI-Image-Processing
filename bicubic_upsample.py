import numpy as np

"""
  Perform cubic interpolation on a 1D array of 4 points.

  Parameters:
  p (numpy.ndarray): Array of 4 points used for interpolation.
  x (float): Fractional distance between the two middle points in the array.

  Returns:
  float: The interpolated value.
  """

def cubic_interpolate(pt, x):
    return pt[1] + 0.5 * x * (pt[2] - pt[0] + x * (
                2.0 * pt[0] - 5.0 * pt[1] + 4.0 * pt[2] - pt[3] + x * (3.0 * (pt[1] - pt[2]) + pt[3] - pt[0])))


def bicubic_interpolate(image, x, y):
    x_int = int(np.floor(x))
    y_int = int(np.floor(y))
    x_fract = x - x_int
    y_fract = y - y_int

    pixels = np.zeros((4, 4))
    for j in range(-1, 3):
        for i in range(-1, 3):
            x_idx = min(max(x_int + i, 0), image.shape[1] - 1)
            y_idx = min(max(y_int + j, 0), image.shape[0] - 1)
            pixels[j + 1, i + 1] = image[y_idx, x_idx]

    col_values = np.zeros(4)
    for j in range(4):
        col_values[j] = cubic_interpolate(pixels[j, :], x_fract)

    return cubic_interpolate(col_values, y_fract)


def bicubic_upsample(image, scale_factor):
    original_height, original_width = image.shape
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    upscaled_image = np.zeros((new_height, new_width))

    for y in range(new_height):
        for x in range(new_width):
            original_x = x / scale_factor
            original_y = y / scale_factor
            upscaled_image[y, x] = bicubic_interpolate(image, original_x, original_y)



    return upscaled_image
