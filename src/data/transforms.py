from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def build_image_transform(size: int = 224, letterbox: bool = False):
    resize = (LetterboxResize(size) if letterbox
              else transforms.Resize((size, size)))
    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform(size: int = 224, augment: bool = False,
                          letterbox: bool = False):
    """Train-time transform with optional pose augmentation.

    For categories whose canonical pose is fixed (e.g. bottle viewed from
    above) augment=False is correct. For categories with natural rotation
    or position variation (e.g. hazelnut), enable augment so the memory
    bank covers those poses; otherwise normal test images at unseen
    rotations score as anomalies.
    """
    if not augment:
        return build_image_transform(size, letterbox=letterbox)
    resize = (LetterboxResize(size) if letterbox
              else transforms.Resize((size, size)))
    return transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180, expand=False,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
        # Mild lighting jitter so the bank covers small exposure / colour
        # variation between train and test batches.
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mask_transform(size: int = 224, letterbox: bool = False):
    resize = (LetterboxResize(size, fill=0) if letterbox
              else transforms.Resize(
                  (size, size),
                  interpolation=transforms.InterpolationMode.NEAREST))
    return transforms.Compose([
        resize,
        transforms.PILToTensor(),
    ])
