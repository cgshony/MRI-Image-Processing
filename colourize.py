import os
import colorsys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pseudocolor(val, minval, maxval):
    h = (float(val - minval) / (maxval - minval)) * 120
    r, g, b = colorsys.hsv_to_rgb(h / 360, 1., 1.)
    return r, g, b

def find_min_max(image):
    minval = np.min(image)
    maxval = np.max(image)
    return minval, maxval

def create_pseudo_color_image(pixels, sizeX, sizeY, minval, maxval):
    im = Image.new(mode="RGB", size=(sizeX, sizeY))
    px = im.load()
    for i in range(sizeX):
        for j in range(sizeY):
            pixel = pixels[j, i]  # Note the swap of indices to match image dimensions
            r, g, b = pseudocolor(pixel, minval, maxval)
            px[i, j] = int(r * 255), int(g * 255), int(b * 255)
    return im

def plot_image(image, title="Image", cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Read a JPEG/PNG image
def read_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

# Convert to grayscale
def convert_to_grayscale(image):
    if image.ndim == 3:
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        grayscale_image = 0.2989 * red + 0.5870 * green + 0.1140 * blue
        return np.round(grayscale_image).astype(np.uint8)
    elif image.ndim == 2:
        return image
    else:
        raise ValueError("Unsupported image dimension")

# File paths (update these to your local paths)
image_path = r"D:\CS\Portfolio\MRI\MRI-Image-Processing\14 no.jpg"

# Process the JPEG/PNG image
if os.path.exists(image_path):
    jpeg_image = read_image(image_path)
    plot_image(jpeg_image, title="Original JPEG Image")
    grayscale_image = convert_to_grayscale(jpeg_image)
    minval, maxval = find_min_max(grayscale_image)
    pseudo_color_image = create_pseudo_color_image(grayscale_image, grayscale_image.shape[1], grayscale_image.shape[0], minval, maxval)
    plot_image(pseudo_color_image, title="Pseudo Color JPEG Image", cmap=None)
