import yaml
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd, Lambdad,
    ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord, MapTransform
)
from monai.transforms import Lambdad, ConcatItemsd, ScaleIntensityRanged
from preprocessing.clahe_transform import ApplyCLAHEd


def load_config(path="config/phe_sich.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_monai_transforms(config_path="config/phe_sich.yaml"):
    config = load_config(config_path)
    spacing = tuple(config["preprocessing"]["spacing"]["target"])
    shape = tuple(config["preprocessing"]["shape"]["output"])
    print(f"[DEBUG] Resize target shape = {shape}, type = {type(shape)}, len = {len(shape)}")
    windows = config["preprocessing"].get("windows", [[0, 150]])
    keys = ["image", "mask"]

    # Generate multi-window transforms
    window_transforms = []
    for i, (w_min, w_max) in enumerate(windows):
        window_transforms.append(
            ScaleIntensityRanged(
                keys=["image"], a_min=w_min, a_max=w_max,
                b_min=0.0, b_max=1.0, clip=True
            )
        )
        window_transforms.append(Lambdad(keys="image", func=lambda x: x.clone()))

    return Compose([
    LoadImaged(keys=keys, ensure_channel_first=True),
    Orientationd(keys, axcodes="RAS"),
    Spacingd(keys, pixdim=spacing, mode=("bilinear", "nearest")),

    # Multi-window slicing
    *window_transforms,
    ConcatItemsd(keys=["image"] * len(windows), name="image"),

    ApplyCLAHEd(keys=["image"]),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ResizeWithPadOrCropd(keys, spatial_size=shape),
    ToTensord(keys)
    ])

    
    
    

