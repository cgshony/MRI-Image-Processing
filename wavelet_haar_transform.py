import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Haar transform functions
def haar_transform_1d(signal):
    length = signal.size // 2
    output = np.zeros_like(signal)
    for i in range(length):
        output[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
        output[length + i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)
    return output

def haar_transform_2d(image):
    rows, cols = image.shape
    transformed_image = np.zeros_like(image, dtype=np.float32)

    # Apply transform to each row
    for i in range(rows):
        transformed_image[i, :] = haar_transform_1d(image[i, :])

    # Apply transform to each column
    for j in range(cols):
        transformed_image[:, j] = haar_transform_1d(transformed_image[:, j])

    return transformed_image

def inverse_haar_transform_1d(transformed_signal):
    length = transformed_signal.size // 2
    output = np.zeros_like(transformed_signal)
    for i in range(length):
        output[2 * i] = (transformed_signal[i] + transformed_signal[length + i]) / np.sqrt(2)
        output[2 * i + 1] = (transformed_signal[i] - transformed_signal[length + i]) / np.sqrt(2)
    return output

def inverse_haar_transform_2d(transformed_image):
    rows, cols = transformed_image.shape
    image = np.zeros_like(transformed_image)

    # Apply inverse Haar transform to each column first
    for j in range(cols):
        image[:, j] = inverse_haar_transform_1d(transformed_image[:, j])

    # Apply inverse Haar transform to each row
    for i in range(rows):
        image[i, :] = inverse_haar_transform_1d(image[i, :])

    return image


# Function to apply enhancement to the high-frequency bands
def enhance_high_frequency_bands(transformed_image, factor=1.5):
    rows, cols = transformed_image.shape
    LL = transformed_image[:rows // 2, :cols // 2]
    LH = transformed_image[:rows // 2, cols // 2:]
    HL = transformed_image[rows // 2:, :cols // 2]
    HH = transformed_image[rows // 2:, cols // 2:]

    # Apply a more subtle enhancement by slightly boosting the high-frequency bands
    LH *= factor
    HL *= factor
    HH *= factor

    # Reconstruct the image by recombining the bands
    transformed_image[:rows // 2, :cols // 2] = LL
    transformed_image[:rows // 2, cols // 2:] = LH
    transformed_image[rows // 2:, :cols // 2] = HL
    transformed_image[rows // 2:, cols // 2:] = HH

    return transformed_image

# Function to plot images
def plot_images(original, transformed, reconstructed, title1="Original", title2="Transformed", title3="Reconstructed"):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(title3)
    plt.axis('off')

    plt.show()

# process the image
def process_image(image):
    # Convert the image to grayscale if it's not already
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Apply Haar transform
    transformed_image = haar_transform_2d(image)

    # Apply enhancement to the high-frequency bands
    enhanced_image = enhance_high_frequency_bands(transformed_image)

    # Apply inverse Haar transform
    reconstructed_image = inverse_haar_transform_2d(enhanced_image)

    # Plot the original, transformed, and reconstructed images
    plot_images(image, transformed_image, reconstructed_image, title1="Original Image", title2="Enhanced Haar Transformed Image", title3="Reconstructed Image")


# Tkinter GUI code
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing with Haar Transform")

        # Main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for image
        self.canvas = tk.Canvas(main_frame, cursor="cross", background="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Control panel
        self.button_frame = tk.Frame(main_frame, bg="lightgray", width=150)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        self.open_button = tk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10, padx=10, anchor="n")

        self.process_button = tk.Button(self.button_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=10, padx=10, anchor="n")

        self.image_path = None
        self.image = None

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.load_image(self.image_path)

    def load_image(self, file_path):
        self.image = io.imread(file_path, as_gray=True)
        self.image = img_as_float(self.image)
        self.display_image(self.image)

    def display_image(self, image):
        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray((image * 255).astype(np.uint8)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def process_image(self):
        if self.image is not None:
            process_image(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
