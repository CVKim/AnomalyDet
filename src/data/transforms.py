from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform(size: int = 224, augment: bool = False):
    """Train-time transform with optional pose augmentation.

    For categories whose canonical pose is fixed (e.g. bottle viewed from
    above) augment=False is correct. For categories with natural rotation
    or position variation (e.g. hazelnut), enable augment so the memory
    bank covers those poses; otherwise normal test images at unseen
    rotations score as anomalies.
    """
    if not augment:
        return build_image_transform(size)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180, expand=False,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mask_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
