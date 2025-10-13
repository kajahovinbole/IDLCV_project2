from torchvision.models import ResNet50_Weights


def get_single_frame_transform(weights=ResNet50_Weights.IMAGENET1K_V1):
    """Bruk denne i Dataset for å matche pretrained-vektene."""
    return weights.transforms()
