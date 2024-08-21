import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

from ScaleImage import scale_image  # Import the scaling function
import bicubic_upsample
import wavelet_haar_transform  # Import Haar transform functions


class ImageProcessing:
    """
    A GUI application for processing MRI images using various image processing techniques,
    including scaling, detail enhancement, and Haar wavelet transforms.
     Attributes:
        root: toplevel window, serves as the main window of the application.
        canvas:  Where the image is displayed.
        ctrl_frame: The frame containing the control buttons.
        rect : The rectangle representing the selected region on the canvas.
        start_x, start_y: The starting x/y-coordinates of the selection rectangle.
        end_x, end_y: The ending coordinates of the selection rectangle.
        image: The currently loaded image.
    """

    def __init__(self, root):

        self.root = root
        self.root.title("MRI Image Processor")

        # Main frame that is the image canvas and the control panel
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the canvas for displaying the image
        self.canvas = tk.Canvas(main_frame, cursor="cross", background="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for control buttons
        self.ctrl_frame = tk.Frame(main_frame, bg="lightgray", width=150)
        self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)


        # Add buttons to the button frame
        self.open_button = tk.Button(self.ctrl_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10, padx=10, anchor="n")

        self.scale_button = tk.Button(self.ctrl_frame, text="Scale Image", command=self.scale_image_button_clicked)
        self.scale_button.pack(pady=10, padx=10, anchor="n")

        self.add_detail_button = tk.Button(self.ctrl_frame, text="Segmentation", command=self.add_detail_button_clicked)
        self.add_detail_button.pack(pady=10, padx=10, anchor="n")

        self.process_button = tk.Button(self.ctrl_frame, text="Process Region", command=self.process_selected_region)
        self.process_button.pack(pady=10, padx=10, anchor="n")

        self.save_button = tk.Button(self.ctrl_frame, text="Save Image", command=self.save_cropped_image)
        self.save_button.pack(pady=10, padx=10, anchor="n")

        #Initialize the region crop area
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.image = None
        self.img_tk = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)


    def open_image(self):
        """
        Opens a file dialog to allow the user to select an image file to open and display.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """Load the selected image file and convert it to grayscale."""

        self.image = Image.open(file_path).convert('L')  # Convert to grayscale
        self.display_image(self.image)

    def display_image(self, image):
        """ Display the selected image on the canvas. """

        image_to_show = image
        self.img_tk = ImageTk.PhotoImage(image_to_show)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def on_button_press(self, event):
        """
        Handles the event when the user presses the mouse button to start selecting a region on the canvas.
        """
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        """
        Handles the event when the user drags the mouse to select a region on the canvas.
        """
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        """
        Handles the event when the user releases the mouse button after selecting a region on the canvas.
        """
        self.end_x = event.x
        self.end_y = event.y

    def crop_selected_region(self):
        """
        Crops the selected region from the currently displayed image.
        """
        if None not in (self.start_x, self.start_y, self.end_x, self.end_y):
            x1 = min(self.start_x, self.end_x)
            y1 = min(self.start_y, self.end_y)
            x2 = max(self.start_x, self.end_x)
            y2 = max(self.start_y, self.end_y)
            cropped_image = self.image.crop((x1, y1, x2, y2))
            return cropped_image
        return None

    def process_selected_region(self):
        """
        Processes the selected region by displaying it in a new window and providing options for further processing.
        """
        cropped_image = self.crop_selected_region()
        if cropped_image:
            self.show_processed_image(cropped_image)

    def show_processed_image(self, cropped_image):
        """
        Displays the cropped image in a new window with options for processing or saving.
        """
        new_window = tk.Toplevel(self.root)
        new_window.title("Cropped Image")

        cropped_image_tk = ImageTk.PhotoImage(cropped_image)
        label = tk.Label(new_window, image=cropped_image_tk)
        label.image = cropped_image_tk
        label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl_frame = tk.Frame(new_window)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)

        process_button = tk.Button(ctrl_frame, text="Process", command=lambda: self.process_cropped_image(cropped_image))
        process_button.pack(pady=10)

        save_button = tk.Button(ctrl_frame, text="Save", command=lambda: self.save_cropped_image(cropped_image))
        save_button.pack(pady=10)

    def save_cropped_image(self, cropped_image=None):

        """
        Save the processed image to a file chosen by the user.
        """

        if cropped_image is None:
            cropped_image = self.image  # Fallback to the current image if no cropped image is provided

        # Open a file dialog for the user to choose the save location and file name
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Image As"
        )

        # Check if the user canceled the save dialog
        if save_path:
            cropped_image.save(save_path)
            print(f"Image saved as {save_path}")
        else:
            print("Save operation canceled.")


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

    def plot_images(self, original, transformed, reconstructed):
        """
        Plots the original, transformed, and reconstructed images side by side.
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


    def scale_image_button_clicked(self):
        """
        Handles the event when the "Scale Image" button is clicked.
        This method scales the loaded image using bicubic interpolation
        with a predefined scale factor. The scaled image is then displayed on the canvas.

        Attributes:
            scale_factor (int): Default is 2.
        """
        if self.image is not None:
            scale_factor = 2  # Base scaling factor
            image_np = np.array(self.image)  # Convert PIL Image to NumPy array
            scaled_image_np = bicubic_upsample.bicubic_upsample(image_np, scale_factor)  # Use the bicubic upsample function
            scaled_image_pil = Image.fromarray(scaled_image_np.astype(np.uint8))  # Convert back to PIL Image
            self.image = scaled_image_pil  # Update the current image
            self.display_image(scaled_image_pil)


    def process_cropped_image(self, cropped_image):
        """
        Processes the cropped image by applying a Haar wavelet transform and displaying the subbands.

        This method normalizes the cropped image, applies a 2D Haar wavelet transform,
        and then plots the resulting LL, LH, HL, and HH subbands.
        """

        cropped_image_np = np.array(cropped_image)

        cropped_image_np = cropped_image_np / 255.0 # Normalize the image to the range [0, 1]

        transformed_image = wavelet_haar_transform.haar_transform_2d(cropped_image_np) # Perform the Haar transform

        # Plot the Haar transform result showing LL, LH, HL, HH subbands
        self.plot_haar_subbands(transformed_image)


    def plot_haar_subbands(self, transformed_image):
        rows, cols = transformed_image.shape
        LL = transformed_image[:rows // 2, :cols // 2]
        LH = transformed_image[:rows // 2, cols // 2:]
        HL = transformed_image[rows // 2:, :cols // 2]
        HH = transformed_image[rows // 2:, cols // 2:]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(LL, cmap='gray')
        axes[0, 0].set_title('LL')
        axes[0, 1].imshow(LH, cmap='gray')
        axes[0, 1].set_title('LH')
        axes[1, 0].imshow(HL, cmap='gray')
        axes[1, 0].set_title('HL')
        axes[1, 1].imshow(HH, cmap='gray')
        axes[1, 1].set_title('HH')
        for ax in axes.flat:
            ax.axis('off')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessing(root)
    root.mainloop()

s

"""Example of a DataFrame which is yet to be implemented to store the data for various images"""
# data = {
#     'Original Image': [image_np],
#     'Transformed Image': [transformed_image],
#     'Processed Image': [processed_image],
#     'Upscaled Image': [upscaled_image],
#     'LL Channel': [LL],
#     'LH Channel (Processed)': [LH_processed],
#     'HL Channel': [HL],
#     'HH Channel': [HH]
# }
#
# df = pd.DataFrame(data)

