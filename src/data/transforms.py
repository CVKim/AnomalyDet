from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mask_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
