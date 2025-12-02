import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from dataset import TrainDataset
from cycleNet.model import create_model, load_state_dict

FILENAMES = set([
    "0000000000-1",
    "0000000001-1",
    "0000000001-1_1",
])
CONFIG_PATH = "./models/cycle_v21.yaml"
CKPT_PATH = "./checkpoints/model_1/step-049999.ckpt"
FIGS_DIR = "./figs/model_1"


def to_device(batch: dict, device: torch.device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def main():
    # ----------
    # Load Model
    # ----------
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(CONFIG_PATH)
    state_dict = load_state_dict(CKPT_PATH, location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    model.eval()

    # ----------
    # Load Test Data
    # ----------
    print("Loading dataset...")
    dataset = TrainDataset()

    indices = []
    for i, item in enumerate(dataset.data):
        stem = Path(item["image"]).stem  # e.g. "0002002383-1"
        if stem in FILENAMES:
            indices.append(i)

    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # ----------
    # Log Images
    # ----------
    print("Testing model...")

    cfg = [1.00,1.50,2.00,2.50,3.00,3.50,4.00,4.50,5.00]

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_device(batch, device)

            for c in cfg:
                out_dir = os.path.join(FIGS_DIR, f"{str(c).replace('.','_')}")
                os.makedirs(out_dir, exist_ok=True)

                if c == 1.00:
                    logs = model.log_images(
                        batch, 
                        split="test", 
                        unconditional_guidance_scale=c,
                        sample=True
                    )
                    key = 'samples'
                else:
                    logs = model.log_images(
                        batch, 
                        split="test", 
                        unconditional_guidance_scale=c
                    )
                    key = f"samples_cfg_scale_{c:.2f}"

                x = logs[key]
                x = x.float().clamp(-1.0, 1.0)
                x = (x + 1.0) / 2.0

                stem = Path(dataset.data[indices[i]]["image"]).stem
                out_path = Path(out_dir) / f"{stem}_{cfg}.tif"
                save_image(x[0].cpu(), out_path)
                print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()