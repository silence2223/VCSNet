"""RGB images with point labels; separate train, validation and test lists."""
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_records(root, list_name):
    """Each nonempty line contains image_path label_path, relative to root."""
    root = Path(root).resolve()
    list_path = root / list_name
    if not list_path.is_file():
        raise FileNotFoundError("Missing split list: " + str(list_path))
    records = []
    seen = set()
    for number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{list_name}:{number}: expected image_path label_path")
        paths = []
        for field in fields:
            relative = Path(field)
            if relative.is_absolute() or ":" in field or "\\" in field:
                raise ValueError("Split entries must use relative paths with forward slashes.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Split entry escapes the dataset root.")
            if not path.is_file():
                raise FileNotFoundError("Missing dataset file: " + str(path))
            paths.append(path)
        if paths[0] in seen:
            raise ValueError("Duplicate image in split: " + fields[0])
        seen.add(paths[0])
        records.append(tuple(paths))
    if not records:
        raise ValueError("Split contains no images: " + list_name)
    return sorted(records, key=lambda row: str(row[0]))


def validate_split_separation(options, split):
    """Reject shared list files and overlapping resolved image paths."""
    if split not in ("train", "val", "test", "background"):
        raise ValueError("Unknown split: " + split)
    root = Path(options["root"]).resolve()
    own_list = options[split + "_list"]
    own = read_records(root, own_list)
    images = {row[0] for row in own}
    for other in ("train", "val", "test"):
        if other == split:
            continue
        other_name = options[other + "_list"]
        if (root / own_list).resolve() == (root / other_name).resolve():
            raise ValueError(f"{split} and {other} must use different list files.")
        if (root / other_name).is_file():
            overlap = images.intersection(row[0] for row in read_records(root, other_name))
            if overlap:
                raise ValueError(f"{split} and {other} overlap in {len(overlap)} image(s).")
    return own


class PointDataset(Dataset):
    def __init__(self, records, transform, train=False):
        self.records = records
        self.img_list = [str(row[0]) for row in records]
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label_path = self.records[index]
        image, points = load_data(image_path, label_path)
        confirmed_blank = len(points) == 0
        image = self.transform(image)
        if self.train:
            scale = random.uniform(0.7, 1.3)
            if scale * min(image.shape[1:]) > 128:
                image = torch.nn.functional.upsample_bilinear(image.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale
            image, points = random_crop(image, points)
            if random.random() > 0.5:
                image = torch.Tensor(image[:, :, :, ::-1].copy())
                for group in points:
                    group[:, 0] = 128 - group[:, 0]
        else:
            # Retain the training implementation's RNG call, without augmenting evaluation.
            random.random()
            points = [points]
        image = torch.Tensor(image)
        targets = []
        for group in points:
            targets.append({"point": torch.as_tensor(group, dtype=torch.float32).reshape(-1, 2),
                            "image_id": torch.tensor([index], dtype=torch.long),
                            "labels": torch.ones(len(group), dtype=torch.long),
                            "is_confirmed_blank": torch.tensor([confirmed_blank], dtype=torch.bool)})
        return image, targets


def load_data(image_path, label_path):
    bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image: " + str(image_path))
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    points = []
    for number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Label line {number}: expected x y")
        point = [float(value) for value in fields]
        if not np.isfinite(point).all():
            raise ValueError("Label coordinates must be finite.")
        points.append(point)
    return image, np.asarray(points, dtype=np.float32).reshape(-1, 2)


def random_crop(image, points, num_patch=4):
    result = np.zeros([num_patch, image.shape[0], 128, 128])
    groups = []
    for index in range(num_patch):
        top = random.randint(0, image.size(1) - 128)
        left = random.randint(0, image.size(2) - 128)
        bottom, right = top + 128, left + 128
        result[index] = image[:, top:bottom, left:right]
        selected = (points[:, 0] >= left) & (points[:, 0] <= right) & (points[:, 1] >= top) & (points[:, 1] <= bottom)
        group = points[selected].copy()
        group[:, 0] -= left
        group[:, 1] -= top
        groups.append(group)
    return result, groups
