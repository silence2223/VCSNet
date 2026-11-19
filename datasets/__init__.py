"""Dataset construction and image preprocessing."""
import torchvision.transforms as transforms
from .point_dataset import PointDataset, validate_split_separation
from .collate import collate_fn_crowd


def image_transform():
    """ImageNet normalization shared with the VCSM RGB reconstruction."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataset(config, split):
    """Use four crops for training and separate native-size validation and test lists."""
    options = config["dataset"]
    records = validate_split_separation(options, split)
    return PointDataset(records, transform=image_transform(), train=split == "train")
