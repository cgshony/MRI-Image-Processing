import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

# from ScaleImage import scale_image  # Import the scaling function
import bicubic_upsample
import wavelet_haar_transform  # Import Haar transform functions


class ImageProcessing:

    def __init__(self, root):

        self.root = root
        self.root.title('MRI')

        main_frame = tk.Frame(root) #local var,only needed to create the frame and pack it
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, cursor="cross", background="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for control buttons
        self.ctrl_frame = tk.Frame(main_frame, bg="lightgray", width=150)
        self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.open_button = tk.Button(self.ctrl_frame, text = 'Open', command=self.open_image)
        self.open_button.pack(pady=10, padx=10, anchor="n")

        # Correct order: create the button first, then pack it
        self.scale_button = tk.Button(self.ctrl_frame, text='Scale', command= self.scale_image_bicubic)
        self.scale_button.pack(pady=10, padx=10, anchor="n")

        self.process_button = tk.Button(self.ctrl_frame, text="Process Region", command=self.add_detail)
        self.process_button.pack(pady=10, padx=10, anchor="n")




    def open_image(self):
        file_path = filedialog.askopenfilename()
        self.load_image(file_path)

    def load_image(self, file_path):
        self.image = Image.open(file_path).convert('L')
        self.display_image(self.image)

    def display_image(self, image):
        image_show = image
        self.img_tk = ImageTk.PhotoImage(image_show)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def scale_image_bicubic(self):
        scale_factor = 2
        scale_image_np = np.array(self.image)
        scaled_image = bicubic_upsample.bicubic_upsample(scale_image_np, scale_factor)
        scaled_image_pil = Image.fromarray(scaled_image.astype(np.uint8))
        self.image = scaled_image_pil
        self.display_image(scaled_image_pil)


    def add_detail(self):

        if self.image is not None:
            image_np = np.array(self.image) / 255.0
            transformed_image = wavelet_haar_transform.haar_transform_2d(image_np)
            reconstructed_image = wavelet_haar_transform.inverse_haar_transform_2d(transformed_image)
            reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  #NB! Convert back to [0, 255], otherwise it breaks


            self.plot_images(self.image, transformed_image, reconstructed_image)

    def plot_images(self, original, transformed, reconstructed):

        fig, axes = plt.subplots(1,3, figsize=(18,6))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
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