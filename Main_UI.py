import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

# from ScaleImage import scale_image  # Import the scaling function
import bicubic_upsample
import wavelet_haar_transform  # Import Haar transform functions

class ImageProcessing:
    """
    A GUI application for processing MRI images using various image processing techniques,
    including scaling, detail enhancement, and Haar wavelet transforms.
     Attributes:
        root (tk.Tk): The root window of the Tkinter application.
        canvas (tk.Canvas):  Where the image is displayed.
        ctrl_frame (tk.Frame): The frame containing the control buttons.
        rect (int): The rectangle representing the selected region on the canvas.
        start_x (int): The starting x-coordinate of the selection rectangle.
        start_y (int): The starting y-coordinate of the selection rectangle.
        end_x (int): The ending x-coordinate of the selection rectangle.
        end_y (int): The ending y-coordinate
        of the selection rectangle.
        image (PIL.Image): The currently loaded image.
        img_tk (ImageTk.PhotoImage): The image object used for displaying on the canvas.
    """

    def __init__(self, root):
        self.root = root
        self.root.title('MRI Image Processor')

        #main frame which is the image cancvas and control panel
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the canvas for displaying the image
        self.canvas = tk.Canvas(main_frame, cursor="cross", background="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(main_frame, bg="lightgray", width=150)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # # Add buttons to the button frame
        self.open_button = tk.Button(self.control_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10, padx=10, anchor="n")

        self.scale_button = tk.Button(self.control_frame, text="Scale Image", command=self.scale_image_button_clicked)
        self.scale_button.pack(pady=10, padx=10, anchor="n")

        self.add_detail_button = tk.Button(self.control_frame, text="Segmentation", command=self.add_detail_button_clicked)
        self.add_detail_button.pack(pady=10, padx=10, anchor="n")

        # self.process_button = tk.Button(self.control_frame, text="Process Region", command=self.process_selected_region)
        # self.process_button.pack(pady=10, padx=10, anchor="n")

        # self.save_button = tk.Button(self.control_frame, text="Save Image", command=self.save_cropped_image)
        # self.save_button.pack(pady=10, padx=10, anchor="n")


    def open_image(self):
        """
         Defines the action taken when the button is clicked - opens a file dialog to allow the user to select an image file to open and display.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """
        Loads the selected image file and converts it to grayscale.
        Args:
        file_path (str): The path to the image file.
        """
        self.image = Image.open(file_path).convert('L')  # Convert to grayscale
        self.display_image(self.image)

    def display_image(self, image):
        """
        Displays the image on the canvas.
        Args:
            image (PIL.Image): The image to display.
        """
        image_to_show = image if isinstance(image, Image.Image) else Image.fromarray((image * 255).astype(np.uint8))
        self.img_tk = ImageTk.PhotoImage(image_to_show)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    """CROP REGION"""

    def crop_selected_region(self):
        pass




    def add_detail_button_clicked(self):
        """
        Enhances the image by applying a Haar wavelet transform and then reconstructing it.
        """
        if self.image is not None:
            image_np = np.array(self.image) / 255.0  # Normalize the image to [0, 1]
            transformed_image = wavelet_haar_transform.haar_transform_2d(image_np)  # Apply Haar transform
            reconstructed_image = wavelet_haar_transform.inverse_haar_transform_2d(
                transformed_image)  # Reconstruct the image
            reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  #NB! Convert back to [0, 255], otherwise it breaks

            # Plot the original, transformed, and reconstructed images
            self.plot_images(self.image, transformed_image, reconstructed_image)



    def scale_image_button_clicked(self):
        """
        Handles the event when the "Scale Image" button is clicked.

        This method scales the loaded image using bicubic interpolation
        with a predefined scale factor. The scaled image is then displayed on the canvas.

        Attributes:
            scale_factor (int): The factor by which the image is scaled. Default is 2.
        """
        if self.image is not None:
            scale_factor = 2  # Base scaling factor
            image_np = np.array(self.image)  # Convert PIL Image to NumPy array
            scaled_image_np = bicubic_upsample.bicubic_upsample(image_np, scale_factor)  # Use the bicubic upsample function
            scaled_image_pil = Image.fromarray(scaled_image_np.astype(np.uint8))  # Convert back to PIL Image
            self.image = scaled_image_pil  # Update the current image
            self.display_image(scaled_image_pil)


    def plot_images(self, original, transformed, reconstructed):
        """
        Plots the original, transformed, and reconstructed images side by side.

        Args:
            original (PIL.Image): The original image.
            transformed (np.ndarray): The transformed image.
            reconstructed (np.ndarray): The reconstructed image after processing.
        """

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(transformed, cmap='gray')
        axes[1].set_title("Haar Transformed Image")
        axes[1].axis('off')

        axes[2].imshow(reconstructed, cmap='gray')
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')

        plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessing(root)
    root.mainloop()
