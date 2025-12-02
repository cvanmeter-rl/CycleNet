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

#NUM_IMAGES = 16  # how many images to test on
BATCH_SIZE = 1
OUTDIR = "./cycle_sd21_single_simple_prompt_frozenSD_MidControlTrue_losses_changed"

CONFIG_PATH = "./models/cycle_v21.yaml"
#CKPT_PATH = "./models/cycle_sd21_single_simple_prompt_frozenSD_allControl.ckpt"
ckpt_paths = [
'stepstep=020699.ckpt',
'stepstep=041399.ckpt',
'stepstep=062099.ckpt',
'stepstep=006899.ckpt',  
'stepstep=027599.ckpt',  
'stepstep=048299.ckpt',  
'stepstep=068999.ckpt',     
'stepstep=013799.ckpt',  
'stepstep=034499.ckpt', 
'stepstep=055199.ckpt',
]


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

    for ckpt in ckpt_paths:

        # make output subdir for this checkpoint
        ckpt_outdir = Path(OUTDIR) / Path(ckpt).stem
        ckpt_outdir.mkdir(parents=True, exist_ok=True)
        # Load model
        print(f"Loading model from {CONFIG_PATH} and {ckpt}")
        model = create_model(CONFIG_PATH).to(device)
        state = load_state_dict(f'/mnt/cyclenet/CycleNet/checkpoints/models/cycle_sd21_single_simple_prompt_frozenSD_MidControlTrue_losses_changed/{ckpt}', location="cpu")
        model.load_state_dict(state)
        model.eval()
        
        cfg = [1.00,1.50,2.00,2.50,3.00]
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                # move tensor fields to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                for c in cfg:
                    if c == 1.00:
                        logs = model.log_images(batch, split="test", unconditional_guidance_scale=c, sample=True)
                        key = 'samples'
                    else:
                        logs = model.log_images(batch, split="test", unconditional_guidance_scale=c)
                        key = f'samples_cfg_scale_{c:.2f}'
                    
                    print(logs.keys())
                    print(key)
            
                    x = logs[key]  # (1, C, H, W), in [-1, 1]
            
                    # to [0,1] for saving
                    x = x.float().clamp(-1.0, 1.0)
                    x = (x + 1.0) / 2.0
            
                    # use the original filename stem for output name
                    stem = Path(dataset.data[indices[idx]]["image"]).stem
                    out_path = ckpt_outdir / f"{c}_{stem}.tif"
                    save_image(x[0].cpu(), out_path)
                    print(f"Saved {out_path}")
    
        print("Done.")

if __name__ == "__main__":
    main() 









