from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import nibabel.orientations as nio


def save_sample(sample, case_id, config):
    save_format = config.get("save_format", "pt")
    output_dir = Path(config["dir"]["output_dir"])
    input_dir = Path(config["dir"]["input_dir"])
    ct_candidates = list((input_dir / "CT").glob(f"{case_id}*.nii*"))

    if not ct_candidates:
        raise FileNotFoundError(f"[!] CT not found for case {case_id}")
    ct_path = ct_candidates[0]
    ct_nib = nib.load(str(ct_path))
    affine = ct_nib.affine
    orientation = nio.aff2axcodes(affine)

    output_dir.mkdir(parents=True, exist_ok=True)

    img = sample["image"]  # [C, D, H, W]
    mask = sample["mask"]  # [1, D, H, W]

    print(f"\n Case: {case_id}")
    print(f"    ▶ Original CT shape     : {ct_nib.shape}")
    print(f"    ▶ Original orientation : {orientation}")
    print(f"    ▶ Original spacing     : {np.abs(np.diag(affine)[:3])}")
    print(f"    ▶ Tensor shape (image) : {img.shape}")
    print(f"    ▶ Tensor shape (mask)  : {mask.shape}")

    # -------- .pt 저장 --------
    if save_format in ["pt", "both"]:
        torch.save({
            "image": img,
            "mask": mask
        }, output_dir / f"{case_id}.pt")
        print(f"    ✔ Saved .pt to {output_dir / f'{case_id}.pt'}")

    # -------- .nii.gz 저장 --------
    if save_format in ["nii", "both"]:
        c = img.shape[0]

        # 이미지 저장
        for i in range(c):
            img_np = img[i].cpu().numpy()            # [D, H, W]
            save_path = output_dir / f"{case_id}_image_win{i}.nii.gz"
            nib.save(nib.Nifti1Image(img_np, affine), save_path)
            print(f"    ✔ Saved image_win{i} to {save_path}")

        # 마스크 저장
        mask_np = mask.squeeze(0).cpu().numpy()         # [D, H, W]
        mask_path = output_dir / f"{case_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_np, affine), mask_path)
        print(f"    ✔ Saved mask to {mask_path}")

    print("Export complete.\n")
    
def compute_signal_quality(image_tensor, mask_tensor):
    """
    image_tensor: torch.Tensor [C, D, H, W] or numpy.ndarray [C, D, H, W]
    mask_tensor: torch.Tensor [1, D, H, W] or numpy.ndarray [1, D, H, W]
    
    Returns: dict with SNR, CNR, status, lesion_voxel_count, bg_voxel_count
    """
    if hasattr(image_tensor, "cpu"):
        image = image_tensor.cpu().numpy()
    else:
        image = image_tensor

    if hasattr(mask_tensor, "cpu"):
        mask = mask_tensor.cpu().numpy()
    else:
        mask = mask_tensor

    image = image[0]  # assume [1, D, H, W]
    mask = mask[0]

    lesion_pixels = image[mask > 0]
    background_pixels = image[mask == 0]

    lesion_voxel_count = lesion_pixels.size
    bg_voxel_count = background_pixels.size

    if lesion_voxel_count == 0 or bg_voxel_count == 0:
        return {
            "SNR": 0.0,
            "CNR": 0.0,
            "status": "poor",
            "lesion_voxels": lesion_voxel_count,
            "background_voxels": bg_voxel_count
        }

    lesion_mean = np.mean(lesion_pixels)
    bg_mean = np.mean(background_pixels)
    bg_std = np.std(background_pixels)

    snr = lesion_mean / (np.std(lesion_pixels) + 1e-8)
    cnr = abs(lesion_mean - bg_mean) / (bg_std + 1e-8)

    status = "good" if snr > 1.0 and cnr > 0.5 else "poor"

    return {
        "SNR": round(float(snr), 4),
        "CNR": round(float(cnr), 4),
        "status": status,
        "lesion_voxels": lesion_voxel_count,
        "background_voxels": bg_voxel_count
    }
