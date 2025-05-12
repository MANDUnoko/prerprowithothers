"""
preprocess_all.py
   - MONAI pipeline으로 전처리
   - .pt (학습), .nii.gz (radiomics) 저장
   - SNR, CNR 기반 품질 평가 후 good/poor 분류 저장
"""
import os
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from monai.data import Dataset

from preprocessing.monai_pipeline import build_monai_transforms
from preprocessing.save_utils import save_sample, compute_signal_quality

def load_config(path="config/phe_sich.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def prepare_dataset(input_dir):
    ct_dir = Path(input_dir) / "CT"
    mask_dir = Path(input_dir) / "MASK"

    data = []
    for ct_file in sorted(ct_dir.glob("*.nii*")):
        case_id = ct_file.stem.split(".")[0]
        mask_file = mask_dir / ct_file.name
        if not mask_file.exists():
            print(f"[!] Mask not found for: {ct_file.name}")
            continue
        data.append({
            "image": str(ct_file),
            "mask": str(mask_file),
            "case_id": case_id
        })
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phe_sich.yaml")
    parser.add_argument("--case", type=str, default="all", help="case_id or 'all'")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = config["dir"]["input_dir"]
    data_list = prepare_dataset(input_dir)

    print("[*] Building MONAI pipeline...")
    transforms = build_monai_transforms()
    dataset = Dataset(data=data_list, transform=transforms)

    print(f"[*] Processing cases...")
    
    selected_cases = dataset if args.case == "all" else [d for d in dataset if d["case_id"] == args.case]
    
    for sample in tqdm(selected_cases):
        case_id = sample["case_id"]

        # 품질 평가
        snr, cnr = compute_signal_quality(sample["image"], sample["mask"])
        quality = "good" if snr > 1.0 and cnr > 0.5 else "poor"

        # config에 따라 저장 경로를 quality별로 분기
        config["dir"]["output_dir"] = str(Path(config["dir"]["base_output"] or "data/processed") / quality)
        save_sample(sample, case_id, config)

    print("[✔] Preprocessing complete.")

if __name__ == "__main__":
    main()
    
# python scripts/preprocess_all.py --case all
# python scripts/preprocess_all.py --case 0001
