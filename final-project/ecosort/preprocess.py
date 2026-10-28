"""Pre-processing sub-system.

Turns a captured photo into the exact input each classifier expects. Both
paths convert to RGB and resize; the difference is only the resolution and the
numeric representation:

* the SVM path needs the hand-crafted 1,900-number measurement vector, built at
  128x128 for comparability with the trained pipeline;
* the CNN path needs a 224x224 float32 array, with MobileNetV2's own scaling
  applied inside the model.

Keeping this logic in one module is what makes the two techniques comparable:
neither classifier does its own ad-hoc resizing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog

from .config import CNN_INPUT_SIZE, SVM_INPUT_SIZE

# Union[...] rather than ``A | B``: this alias is evaluated at import time, and
# the ``|`` form raises TypeError on Python 3.9. The annotation form is fine
# either way, but this line is an assignment, not an annotation.
ImageSource = Union[bytes, bytearray, str, Path, Image.Image, np.ndarray]

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InvalidImageError(ValueError):
    """Raised when a captured item cannot be decoded into a usable image."""


def load_image(source: ImageSource) -> Image.Image:
    """Decode any supported source into an RGB :class:`PIL.Image.Image`.

    Accepts raw bytes (an upload or a webcam frame), a path, an existing PIL
    image, or a numpy array such as an OpenCV ``BGR``/RGB frame.
    """
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(source)))
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise InvalidImageError(f"No such image file: {path}")
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise InvalidImageError(
                    f"Unsupported image type '{path.suffix}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
                )
            image = Image.open(path)
        elif isinstance(source, np.ndarray):
            array = source
            if array.ndim == 2:
                image = Image.fromarray(array.astype("uint8"), mode="L")
            elif array.ndim == 3 and array.shape[2] == 3:
                image = Image.fromarray(array.astype("uint8"), mode="RGB")
            else:
                raise InvalidImageError(f"Unsupported array shape {array.shape}; expected HxWx3")
        else:
            raise InvalidImageError(f"Unsupported image source type: {type(source).__name__}")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The captured data is not a readable image.") from exc

    return image.convert("RGB")


#: Centre-crop scales used by robust mode. Each view is resized back to the
#: model input, so the object is presented at several apparent sizes.
ROBUST_SCALES: tuple[float, ...] = (1.0, 0.8, 0.6)


def centre_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Crop the middle ``fraction`` of the frame, keeping the subject centred."""
    if fraction >= 1.0:
        return image
    width, height = image.size
    new_width, new_height = int(width * fraction), int(height * fraction)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


def trim_frame(image: Image.Image, max_fraction: float = 0.15) -> Image.Image:
    """Remove a uniform border left by a screenshot, app chrome or a black matte.

    A phone photo has no such border, but a pasted screenshot keeps its window
    furniture, and a dark frame around the item is enough to shift the classifier:
    on a clear glass photographed against white, the screenshot border moved the
    prediction from glass to metal.

    Only edge lines whose mean tone differs sharply from the image median are
    removed, and never more than ``max_fraction`` from any side, so a real photo in
    which the item touches the edge is left alone.
    """
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)
    height, width = array.shape[:2]
    grey = array.mean(axis=2)
    median = np.median(grey)
    threshold = 45.0                      # a strong, unambiguous difference

    def strip_count(lines: np.ndarray) -> int:
        count = 0
        for value in lines:
            if abs(value - median) > threshold:
                count += 1
            else:
                break
        return count

    limit_x = max(1, int(width * max_fraction))
    limit_y = max(1, int(height * max_fraction))
    top = min(strip_count(grey.mean(axis=1)), limit_y)
    bottom = min(strip_count(grey.mean(axis=1)[::-1]), limit_y)
    left = min(strip_count(grey.mean(axis=0)), limit_x)
    right = min(strip_count(grey.mean(axis=0)[::-1]), limit_x)

    if not any((top, bottom, left, right)):
        return rgb
    return rgb.crop((left, top, width - right, height - bottom))


def prepare_for_cnn(image: Image.Image, size: int = CNN_INPUT_SIZE) -> np.ndarray:
    """Resize for the CNN and return one ``(size, size, 3)`` float32 batch entry.

    Scaling (``preprocess_input``) is applied inside the Keras model, so this
    function only resizes. Doing it once, in one place, keeps live inference
    identical to training.
    """
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype="float32")


def prepare_for_svm(image: Image.Image, size: int = SVM_INPUT_SIZE) -> Image.Image:
    """Resize for the hand-crafted feature path."""
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def extract_features(image: Image.Image, size: int = SVM_INPUT_SIZE) -> np.ndarray:
    """Build the 1,900-number measurement vector for the SVM.

    Three families of measurements, exactly as described in the report:

    * HOG (histogram of oriented gradients) over 16x16 cells and 2x2 blocks,
      which captures the shape of the outline;
    * a 32-bin colour histogram per RGB channel, which captures colour;
    * GLCM texture properties (contrast, dissimilarity, homogeneity, energy,
      correlation) at distances 1 and 2 and four angles, which capture surface
      roughness and separate cardboard from plastic film.

    This function must stay byte-for-byte equivalent to the version used during
    training: the SVM was fitted on these exact measurements, so any change here
    silently degrades every prediction.
    """
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb)
    grey = (rgb2gray(array) * 63).astype(np.uint8)  # quantise to 64 levels -> fast GLCM

    hog_features = hog(
        grey,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    histograms = np.concatenate(
        [np.histogram(array[:, :, channel].ravel(), bins=32, range=(0, 256))[0]
         for channel in range(3)]
    )
    histograms = histograms / (histograms.sum() + 1e-6)

    matrix = graycomatrix(
        grey,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    texture = np.concatenate(
        [graycoprops(matrix, prop).ravel() for prop in
         ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]]
    )

    return np.concatenate([hog_features, histograms, texture])


FEATURE_VECTOR_LENGTH = 1900
