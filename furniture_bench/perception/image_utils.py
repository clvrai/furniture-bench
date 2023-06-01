import cv2

from furniture_bench.config import config


def resize(img):
    """Resizes `img` into 224x224."""
    size = config["furniture"]["env_img_size"]
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def resize_crop(img):
    """Resizes `img` and center crops into 224x224."""
    size = config["furniture"]["env_img_size"]
    assert size[0] == size[1]

    # Resize.
    img_size = img.shape
    ratio = 256 / min(img_size[0], img_size[1])
    ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
    img = cv2.resize(img, ratio_size, interpolation=cv2.INTER_AREA)

    # Center crop.
    y = img.shape[0] / 2 - size[0] / 2
    x = img.shape[1] / 2 - size[1] / 2
    crop_img = img[int(y) : int(y + size[0]), int(x) : int(x + size[1])]
    return crop_img
