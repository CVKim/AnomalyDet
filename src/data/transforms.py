from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RoiCropFromBlack:
    """Crop the image to the bounding box of its non-black region.

    Computes a binary mask from the grayscale (`> threshold`), finds the
    tight bounding box of the True pixels, expands it by `margin`, and
    crops to that box. Used for line-scan style sources where the part
    is a thin vertical strip on a black background — cropping first
    means the subsequent resize captures the part at ~3x higher effective
    resolution than resizing the full frame.

    Deterministic on identical input pixels, so the runner can apply the
    same crop to its full-resolution `orig` array independently.
    """

    def __init__(self, threshold: int = 8, margin: int = 16):
        self.threshold = int(threshold)
        self.margin = int(margin)

    def __call__(self, img: Image.Image) -> Image.Image:
        import numpy as np
        arr = np.asarray(img.convert('L'))
        ys, xs = np.where(arr > self.threshold)
        if xs.size == 0 or ys.size == 0:
            return img
        h, w = arr.shape
        x0 = max(0, int(xs.min()) - self.margin)
        x1 = min(w, int(xs.max()) + 1 + self.margin)
        y0 = max(0, int(ys.min()) - self.margin)
        y1 = min(h, int(ys.max()) + 1 + self.margin)
        return img.crop((x0, y0, x1, y1))


def roi_bbox_from_image(arr, threshold: int = 8, margin: int = 16):
    """Same bbox logic as RoiCropFromBlack, but on a numpy array. Returns
    (x0, y0, x1, y1) for downstream callers (e.g. the runner cropping
    its own `orig` copy to match the dataset transform).
    """
    import numpy as np
    if arr.ndim == 3:
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    ys, xs = np.where(gray > int(threshold))
    if xs.size == 0 or ys.size == 0:
        h, w = gray.shape
        return (0, 0, w, h)
    h, w = gray.shape
    m = int(margin)
    x0 = max(0, int(xs.min()) - m)
    x1 = min(w, int(xs.max()) + 1 + m)
    y0 = max(0, int(ys.min()) - m)
    y1 = min(h, int(ys.max()) + 1 + m)
    return (x0, y0, x1, y1)


class LetterboxResize:
    """Resize so the longest side equals `size`, then pad the short side
    with black to a square. Preserves aspect ratio.

    Use for non-square sources (e.g. 4096x2851 line-scan style) where the
    default `Resize((size, size))` would squash the geometry.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # pad to (size, size); black borders end up below the bg-mask
        # threshold, so the FoV-mask post-process will zero them out.
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return transforms.functional.pad(
            img, [left, top, right, bottom], fill=self.fill)


def build_image_transform(size: int = 224, letterbox: bool = False,
                          roi_crop: bool = False, roi_threshold: int = 8,
                          roi_margin: int = 16):
    steps = []
    if roi_crop:
        steps.append(RoiCropFromBlack(threshold=roi_threshold, margin=roi_margin))
    steps.append(LetterboxResize(size) if letterbox
                 else transforms.Resize((size, size)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_train_transform(size: int = 224, augment: bool = False,
                          letterbox: bool = False,
                          roi_crop: bool = False, roi_threshold: int = 8,
                          roi_margin: int = 16):
    """Train-time transform with optional pose augmentation.

    For categories whose canonical pose is fixed (e.g. bottle viewed from
    above) augment=False is correct. For categories with natural rotation
    or position variation (e.g. hazelnut), enable augment so the memory
    bank covers those poses; otherwise normal test images at unseen
    rotations score as anomalies.
    """
    if not augment:
        return build_image_transform(size, letterbox=letterbox,
                                      roi_crop=roi_crop,
                                      roi_threshold=roi_threshold,
                                      roi_margin=roi_margin)
    steps = []
    if roi_crop:
        steps.append(RoiCropFromBlack(threshold=roi_threshold, margin=roi_margin))
    steps.append(LetterboxResize(size) if letterbox
                 else transforms.Resize((size, size)))
    steps += [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180, expand=False,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
        # Mild lighting jitter so the bank covers small exposure / colour
        # variation between train and test batches.
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def build_mask_transform(size: int = 224, letterbox: bool = False):
    resize = (LetterboxResize(size, fill=0) if letterbox
              else transforms.Resize(
                  (size, size),
                  interpolation=transforms.InterpolationMode.NEAREST))
    return transforms.Compose([
        resize,
        transforms.PILToTensor(),
    ])
