import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class TrainDataset(Dataset):
    def __init__(self):
      self.data = []
      self.data_dir = Path('/mnt/synrs3d/SynRS3D/data')
      self.synthetic_dataset_names = [
        "terrain_g05_mid_v1",
        "grid_g05_mid_v2",
        "terrain_g05_low_v1",
        "terrain_g05_high_v1",
        "terrain_g005_mid_v1",
        "terrain_g005_low_v1",
        "grid_g005_mid_v2",
        "terrain_g005_high_v1",
        "terrain_g1_mid_v1",
        "terrain_g1_low_v1",
        "terrain_g1_high_v1",
        "grid_g005_mid_v1",
        "grid_g005_low_v1",
        "grid_g005_high_v1",
        "grid_g05_mid_v1",
        "grid_g05_low_v1",
        "grid_g05_high_v1",
    ]
    for d in self.synthetic_dataset_names:
      
