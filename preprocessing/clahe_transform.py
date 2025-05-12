import torch
import numpy as np
from monai.transforms import MapTransform
import cv2

class ApplyCLAHEd(MapTransform):
    def __init__(self, keys, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__(keys)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)
            if is_tensor:
                arr = arr.detach().cpu().numpy()  # [C, D, H, W]

            enhanced = []
            for c in range(arr.shape[0]):  # 채널별
                channel = []
                for slice_2d in arr[c]:  # 각 2D 슬라이스
                    normed = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                    processed = clahe.apply(normed)
                    channel.append(processed.astype(np.float32) / 255.0)
                enhanced.append(np.stack(channel))  # [D, H, W]

            enhanced_np = np.stack(enhanced)  # [C, D, H, W]
            d[key] = torch.from_numpy(enhanced_np).float() if is_tensor else enhanced_np
        return d
