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


def haar_transform_2d_multilevel(
    image: np.ndarray, levels: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Recursively apply the 2D Haar transform, mip-map/pyramid style: level 1
    decomposes the whole image, level 2 decomposes level 1's LL (approximation)
    quadrant, level 3 decomposes level 2's LL quadrant, and so on - each level
    isolates a coarser scale of detail, the same idea as a Laplacian pyramid
    or a mip-chain in 3D.

    Coefficients are packed into one array the same shape as `image`, exactly
    like `haar_transform_2d` packs a single level: level *i*'s detail bands
    live in the quadrants of the (rows, cols) region returned as sizes[i-1],
    and the next level's LL lives in that region's top-left quadrant.

    Stops early (using fewer than `levels` levels) once the active region
    would drop below 2x2 - there's nothing left to split. Returns the packed
    coefficients plus the list of active region sizes, one per level actually
    applied, needed to locate each level's bands and to invert the transform.
    """
    rows, cols = image.shape
    result = image.astype(np.float32).copy()
    sizes: list[tuple[int, int]] = []
    r, c = rows, cols
    for _ in range(max(levels, 0)):
        if r < 2 or c < 2:
            break
        result[:r, :c] = haar_transform_2d(result[:r, :c])
        sizes.append((r, c))
        r, c = r // 2, c // 2
    return result, sizes


def inverse_haar_transform_2d_multilevel(
    coeffs: np.ndarray, sizes: list[tuple[int, int]]
) -> np.ndarray:
    """Undo `haar_transform_2d_multilevel`: invert each level's transform in
    reverse order (coarsest level first), same pattern as collapsing a
    pyramid back down from its top."""
    result = coeffs.copy()
    for r, c in reversed(sizes):
        result[:r, :c] = inverse_haar_transform_2d(result[:r, :c])
    return result


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


def _nonlinear_detail_gain(band: np.ndarray, factor: float) -> np.ndarray:
    """Boost detail coefficients with a compressive, edge-aware curve instead
    of a flat linear multiply - borrowed from the same idea behind Local
    Laplacian Filters (Paris, Hasinoff & Kautz 2011): treat small-magnitude
    coefficients (subtle texture) differently from large-magnitude ones
    (strong edges), rather than scaling both by the same amount.

    Coefficients are normalized to [-1, 1] by the band's own peak magnitude,
    then remapped by `|x| ** (1/factor)`. For `factor > 1` that exponent is
    < 1, which pulls small `|x|` up proportionally more than large `|x|`
    (whose values are already close to 1 and change little) - so faint
    detail gets boosted while strong edges are only gently touched, instead
    of being amplified outright into ringing/halos. `factor == 1.0` is a
    no-op; `0 < factor < 1` mirrors the same curve the other way, damping
    detail. The result is rescaled back to the band's original range, so
    (unlike a plain multiply) it can't blow past the strongest coefficient
    already present.
    """
    scale = float(np.abs(band).max())
    if scale < 1e-6:
        return band

    factor = max(factor, 1e-3)
    normalized = band / scale
    exponent = 1.0 / factor
    boosted = np.sign(normalized) * np.abs(normalized) ** exponent
    return boosted * scale


def enhance_pyramid(
    image: np.ndarray, factor: float = 1.5, levels: int = 3
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Multi-level counterpart to `enhance_high_frequency_bands`: decompose
    `image` into up to `levels` pyramid levels via
    `haar_transform_2d_multilevel`, then apply `_nonlinear_detail_gain` to
    every level's LH/HL/HH bands (each level gets its own independent gain,
    same `factor`). Returns the enhanced coefficients plus the sizes list
    `inverse_haar_transform_2d_multilevel` needs to invert them.
    """
    coeffs, sizes = haar_transform_2d_multilevel(image, levels)

    for r, c in sizes:
        half_r, half_c = r // 2, c // 2
        lh = coeffs[:half_r, half_c:c]
        hl = coeffs[half_r:r, :half_c]
        hh = coeffs[half_r:r, half_c:c]

        coeffs[:half_r, half_c:c] = _nonlinear_detail_gain(lh, factor)
        coeffs[half_r:r, :half_c] = _nonlinear_detail_gain(hl, factor)
        coeffs[half_r:r, half_c:c] = _nonlinear_detail_gain(hh, factor)

    return coeffs, sizes


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


def build_enhanced_channels(
    image: np.ndarray, factor: float = 1.5, levels: int = 3
) -> list[tuple[str, str, np.ndarray]]:
    """Haar-transform `image` through up to `levels` pyramid levels, boost
    every level's high-frequency bands by `factor` via `_nonlinear_detail_gain`,
    and return every channel worth looking at: the inverse-transformed
    (reconstructed) image, the coarsest level's approximation band, and each
    level's 3 detail sub-bands on their own, normalized for display. List
    order is the slider order in the UI - finest level's details first.

    `levels` is a request, not a guarantee: `haar_transform_2d_multilevel`
    stops early once the active region drops below 2x2, so a small source
    image may end up with fewer levels than asked for.
    """
    enhanced, sizes = enhance_pyramid(image, factor, levels)
    reconstructed = inverse_haar_transform_2d_multilevel(enhanced, sizes)

    channels: list[tuple[str, str, np.ndarray]] = [
        ("reconstructed", "Reconstructed", np.clip(reconstructed, 0, 255))
    ]

    # The coarsest level's LL quadrant is the only one that's a meaningful
    # approximation image on its own - every other level's "LL" region is
    # just the next level's input, already decomposed further.
    last_r, last_c = sizes[-1]
    half_r, half_c = last_r // 2, last_c // 2
    ll = enhanced[:half_r, :half_c]
    channels.append(("ll", "LL - Approximation", _normalize_band(ll, centered=False)))

    for level_num, (r, c) in enumerate(sizes, start=1):
        half_r, half_c = r // 2, c // 2
        lh = enhanced[:half_r, half_c:c]
        hl = enhanced[half_r:r, :half_c]
        hh = enhanced[half_r:r, half_c:c]
        channels.append(
            (f"lh_{level_num}", f"LH L{level_num} - Horizontal detail", _normalize_band(lh, centered=True))
        )
        channels.append(
            (f"hl_{level_num}", f"HL L{level_num} - Vertical detail", _normalize_band(hl, centered=True))
        )
        channels.append(
            (f"hh_{level_num}", f"HH L{level_num} - Diagonal detail", _normalize_band(hh, centered=True))
        )

    return channels


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
