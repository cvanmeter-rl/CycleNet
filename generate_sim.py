import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import TrainDataset
from cycleNet.model import create_model, load_state_dict

SIM_DIR = "/mnt/project/data/sim/model_1/"
CONFIG_PATH = "./models/cycle_v21.yaml"
CKPT_PATH = "./checkpoints/model_1/step-019999.ckpt"

# Define batch size here
BATCH_SIZE = 4

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

    # INDICES are stored here as a torch tensor
    indices = torch.randperm(len(sim_dataset))[:2048]
    subset = Subset(sim_dataset, indices)
    
    # Enable shuffle=False so our index tracking remains deterministic relative to the subset
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
    print(f"Testing model with Batch Size {BATCH_SIZE}...")
    os.makedirs(SIM_DIR, exist_ok=True)

    cfg = 1.0

    with torch.no_grad():
        # Enumerate gives us the batch_idx (0, 1, 2...)
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            batch = to_device(batch, device)

            logs = model.log_images(
                batch,
                split="test",
                unconditional_guidance_scale=cfg,
                sample=True
            )
            
            # x is now shape [BATCH_SIZE, C, H, W]
            x = logs["samples"]
            x = x.float().clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0
            
            # Move entire batch to CPU once to avoid multiple GPU-CPU transfers inside the loop
            x_cpu = x.cpu()

            # Calculate where this batch starts in the global indices list
            batch_start_idx = batch_idx * BATCH_SIZE

            # Loop through the images within this specific batch
            # We use x.size(0) because the very last batch might be smaller than BATCH_SIZE
            for k in range(x.size(0)):
                
                # 1. Get the global index for this specific image
                global_idx = batch_start_idx + k
                
                # 2. Retrieve the original dataset index from your random permutation
                # .item() converts the 0-d tensor to a standard Python integer
                original_dataset_idx = indices[global_idx].item()
                
                # 3. Get the filename using the original index
                stem = Path(sim_dataset.data[original_dataset_idx]["image"]).stem
                out_path = Path(SIM_DIR) / f"{stem}_{cfg}.tif"
                
                # 4. Save the single image k
                save_image(x_cpu[k], out_path)
                print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()