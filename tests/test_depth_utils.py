import numpy as np

from tips_decoder.utils import depth as depth_utils


def test_normalize_depth_range():
    depth = np.array([[0.0, 2.0], [4.0, 6.0]], dtype="float32")
    normalized = depth_utils.normalize_depth(depth)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_depth_to_grayscale_size():
    depth = np.random.rand(4, 4).astype("float32")
    image = depth_utils.depth_to_grayscale(depth)
    assert image.size == (4, 4)


def test_depth_resize_shape():
    depth = np.random.rand(4, 6).astype("float32")
    resized = depth_utils.resize_depth(depth, (8, 10))
    assert resized.shape == (10, 8)
