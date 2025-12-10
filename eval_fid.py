import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from dataset import TrainDataset
from cycleNet.model import create_model, load_state_dict

REAL_DIR = "/mnt/project/data/real/all"
SIM_DIR = "/mnt/project/data/sim/all"
CONFIG_PATH = "./models/custom/real.yaml"
CKPT_PATH = "./runs/real/checkpoints/step-009999.ckpt"


def to_device(batch: dict, device: torch.device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def main():
    # ----------
    # Initialize CycleNet model
    # ----------
    print("Initializing model architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(CONFIG_PATH)
    model = model.to(device)
    model.eval()

    # ----------
    # Load TrainDataset test images
    # ----------
    print("Loading dataset...")
    sim_dataset = TrainDataset(include_real=False)

    indices = torch.randperm(len(sim_dataset))[:2048]
    subset = Subset(sim_dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # ----------
    # Load Model
    # ----------
    print(f"\n---- Testing {CKPT_PATH} ----")

    print("Loading model...")
    state_dict = load_state_dict(CKPT_PATH, location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # ----------
    # Log Images
    # ----------
    print("Testing model...")
    os.makedirs(SIM_DIR, exist_ok=True)

    cfg = 1.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_device(batch, device)

            logs = model.log_images(
                batch,
                split="test",
                unconditional_guidance_scale=cfg,
                sample=True
            )
            
            x = logs["samples"]
            x = x.float().clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0

            stem = Path(sim_dataset.data[indices[i]]["image"]).stem
            out_path = Path(SIM_DIR) / f"{stem}_{cfg}.tif"
            save_image(x[0].cpu(), out_path)
            print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
