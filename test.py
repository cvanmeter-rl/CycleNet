import os
from pathlib import Path

#import torch
#from torch.utils.data import DataLoader, Subset
#from torchvision.utils import save_image

from dataset import TrainDataset
#from cycleNet.model import create_model, load_state_dict

FILENAMES = [
    "0000000000-1",
    "0000000001-1",
    "0000000001-1",
]

NUM_IMAGES = 16  # how many images to test on
BATCH_SIZE = 1
OUTDIR = "./test_outputs"

CONFIG_PATH = "./models/cycle_v21.yaml"
CKPT_PATH = "./models/cycle_sd21_single_simple_prompt_frozenSD_allControl.ckpt"


def main():
  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(OUTDIR, exist_ok=True)

  dataset = TrainDataset()
  # num_to_use = min(NUM_IMAGES, len(dataset))
  # subset = Subset(dataset, list(range(num_to_use)))

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

  









