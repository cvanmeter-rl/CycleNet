import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from dataset import TrainDataset
from cycleNet.model import create_model, load_state_dict

FILENAMES = [
    "0000000000-1",
    "0000000001-1",
    "0000000001-1_1",
]

NUM_IMAGES = 16  # how many images to test on
BATCH_SIZE = 1
OUTDIR = "./test_outputs_3"

CONFIG_PATH = "./models/cycle_v21.yaml"
CKPT_PATH = "./models/cycle_sd21_single_simple_prompt_frozenSD_allControl.ckpt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTDIR, exist_ok=True)
    
    dataset = TrainDataset()

    wanted = set(FILENAMES)
    indices = []
    for i, item in enumerate(dataset.data):
        stem = Path(item["image"]).stem  # e.g. "0002002383-1"
        if stem in wanted:
            indices.append(i)

    print(f"Requested {len(FILENAMES)} filenames, found {len(indices)} in dataset.")
    if not indices:
        print("No matching files found — check FILENAMES list.")
        return

    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # Load model
    print(f"Loading model from {CONFIG_PATH} and {CKPT_PATH}")
    model = create_model(CONFIG_PATH).to(device)
    state = load_state_dict(CKPT_PATH, location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # move tensor fields to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
    
            logs = model.log_images(batch, split="test")
    
            if "samples" not in logs:
                print(f"'samples' not in log_images keys: {list(logs.keys())}")
                return
    
            x = logs["samples"]  # (1, C, H, W), in [-1, 1]
    
            # to [0,1] for saving
            x = x.float().clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0
    
            # use the original filename stem for output name
            stem = Path(dataset.data[indices[idx]]["image"]).stem
            out_path = Path(OUTDIR) / f"{stem}_cyclenet.tif"
            save_image(x[0].cpu(), out_path)
            print(f"Saved {out_path}")

    print("Done.")

if __name__ == "__main__":
    main() 









