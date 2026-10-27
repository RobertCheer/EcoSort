"""Pre-processing tests.

The SVM was fitted on a specific measurement vector, so these tests pin the
things that would silently invalidate every prediction if they changed: the
vector length, the dtype, the determinism, and the resize target.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ecosort.config import CNN_INPUT_SIZE, SVM_INPUT_SIZE
from ecosort.preprocess import (
    FEATURE_VECTOR_LENGTH,
    InvalidImageError,
    centre_crop,
    extract_features,
    load_image,
    prepare_for_cnn,
    prepare_for_svm,
)


@pytest.fixture()
def rgb_image() -> Image.Image:
    return Image.new("RGB", (512, 384), (120, 160, 90))


@pytest.fixture()
def png_bytes(rgb_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------- loading


def test_load_from_bytes(png_bytes):
    image = load_image(png_bytes)
    assert image.mode == "RGB"
    assert image.size == (512, 384)


def test_load_from_path(tmp_path, rgb_image):
    path = tmp_path / "item.png"
    rgb_image.save(path)
    assert load_image(path).size == (512, 384)


def test_load_from_pil_passthrough(rgb_image):
    assert load_image(rgb_image).size == rgb_image.size


def test_load_from_array():
    array = np.zeros((64, 48, 3), dtype="uint8")
    image = load_image(array)
    assert image.size == (48, 64)
    assert image.mode == "RGB"


def test_load_converts_rgba_to_rgb():
    rgba = Image.new("RGBA", (10, 10), (1, 2, 3, 4))
    assert load_image(rgba).mode == "RGB"


def test_load_rejects_garbage_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"this is not an image")


def test_load_rejects_missing_path(tmp_path):
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "nope.png")


def test_load_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "item.tiff"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(InvalidImageError):
        load_image(path)


def test_load_rejects_unsupported_type():
    with pytest.raises(InvalidImageError):
        load_image(12345)  # type: ignore[arg-type]  # deliberately wrong type


def test_load_rejects_bad_array_shape():
    with pytest.raises(InvalidImageError):
        load_image(np.zeros((4, 4, 4, 4), dtype="uint8"))


# ---------------------------------------------------------------- CNN prepare


def test_cnn_prepare_shape_and_dtype(rgb_image):
    array = prepare_for_cnn(rgb_image)
    assert array.shape == (CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3)
    assert array.dtype == np.float32


def test_cnn_prepare_preserves_raw_pixel_range(rgb_image):
    """Scaling belongs to the model, so raw 0-255 values must survive here."""
    array = prepare_for_cnn(rgb_image)
    assert array.max() <= 255.0
    assert array.min() >= 0.0


def test_cnn_prepare_handles_non_square_input():
    wide = Image.new("RGB", (900, 200), (10, 20, 30))
    assert prepare_for_cnn(wide).shape == (CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3)


# ---------------------------------------------------------------- SVM features


def test_svm_prepare_resizes(rgb_image):
    assert prepare_for_svm(rgb_image).size == (SVM_INPUT_SIZE, SVM_INPUT_SIZE)


def test_feature_vector_length(rgb_image):
    """A length change invalidates the trained pipeline, so it is pinned."""
    features = extract_features(rgb_image)
    assert features.shape == (FEATURE_VECTOR_LENGTH,)
    assert features.ndim == 1


def test_features_are_finite(rgb_image):
    assert np.all(np.isfinite(extract_features(rgb_image)))


def test_features_are_deterministic(rgb_image):
    assert np.array_equal(extract_features(rgb_image), extract_features(rgb_image))


def test_features_are_resolution_independent(rgb_image):
    """The extractor resizes internally, so input size must not matter."""
    large = rgb_image.resize((1024, 768))
    assert extract_features(large).shape == extract_features(rgb_image).shape


def test_features_differ_between_images():
    a = Image.new("RGB", (128, 128), (0, 0, 0))
    b = Image.new("RGB", (128, 128), (255, 255, 255))
    assert not np.allclose(extract_features(a), extract_features(b))


def test_features_handle_grayscale_input():
    grey = Image.new("L", (100, 100), 128)
    assert extract_features(grey).shape == (FEATURE_VECTOR_LENGTH,)


def test_centre_crop_keeps_the_middle_and_respects_full_frame():
    """Robust mode builds its views with this, so the geometry must be exact."""
    image = Image.new("RGB", (100, 60), (10, 20, 30))

    half = centre_crop(image, 0.5)
    assert half.size == (50, 30)
    assert half.getpixel((0, 0)) == (10, 20, 30)

    assert centre_crop(image, 1.0) is image
