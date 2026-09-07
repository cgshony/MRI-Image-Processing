import numpy as np
import matplotlib.pyplot as plt

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


def _normalize_band(band: np.ndarray, *, centered: bool) -> np.ndarray:
    """Rescale a Haar-domain band to a displayable 0-255 range.

    Detail bands (LH/HL/HH) are coefficients centered on zero, where the sign
    carries information (the direction of an edge) - `centered=True` maps 0 to
    mid-gray (128) and scales symmetrically by the largest-magnitude
    coefficient, so edges of both polarities stay visible. The approximation
    band (LL) is a plain (if rescaled) intensity - `centered=False` does a
    standard min-max stretch instead.
    """
    if centered:
        scale = float(np.abs(band).max())
        if scale < 1e-6:
            return np.full_like(band, 128.0)
        return band / scale * 127.0 + 128.0

    lo, hi = float(band.min()), float(band.max())
    if hi - lo < 1e-6:
        return np.zeros_like(band)
    return (band - lo) / (hi - lo) * 255.0


def build_enhanced_channels(image: np.ndarray, factor: float = 1.5) -> list[tuple[str, str, np.ndarray]]:
    """Haar-transform `image`, boost its high-frequency bands by `factor`, and
    return every channel worth looking at: the inverse-transformed
    (reconstructed) image plus each of the 4 Haar sub-bands on its own,
    normalized for display. List order is the slider order in the UI.
    """
    transformed = haar_transform_2d(image)
    enhanced = enhance_high_frequency_bands(transformed, factor)
    reconstructed = inverse_haar_transform_2d(enhanced)

    rows, cols = enhanced.shape
    half_r, half_c = rows // 2, cols // 2
    ll = enhanced[:half_r, :half_c]
    lh = enhanced[:half_r, half_c:]
    hl = enhanced[half_r:, :half_c]
    hh = enhanced[half_r:, half_c:]

    return [
        ("reconstructed", "Reconstructed", np.clip(reconstructed, 0, 255)),
        ("ll", "LL - Approximation", _normalize_band(ll, centered=False)),
        ("lh", "LH - Horizontal detail", _normalize_band(lh, centered=True)),
        ("hl", "HL - Vertical detail", _normalize_band(hl, centered=True)),
        ("hh", "HH - Diagonal detail", _normalize_band(hh, centered=True)),
    ]


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
