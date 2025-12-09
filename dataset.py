import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path

class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.data_dir = Path("/mnt/project/data/")
        self.source = "synthetic satellite map render, flat colors, minimal texture, clean computer-generated style, no haze, no noise, no shadows"
        self.target = "realistic high-resolution satellite image, natural colors, rich textures, realistic lighting, sensor noise, atmospheric haze, soft shadows"
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
        self.real_dataset_names = [
            "real/DFC18/DFC18",
            "real/DFC19",
            "real/geonrw/data"
        ]
        # -- Missing images to ignore
        self.missing_files = set([
            '/mnt/project/data/grid_g05_mid_v2/opt/0000449416-1.tif', 
            '/mnt/project/data/grid_g005_mid_v2/opt/0000069746-1.tif', 
            '/mnt/project/data/grid_g005_mid_v1/opt/0000008541-1.tif', 
            '/mnt/project/data/grid_g005_low_v1/opt/0000335672-1.tif', 
            '/mnt/project/data/grid_g005_high_v1/opt/0000301247-1.tif', 
            '/mnt/project/data/grid_g05_mid_v1/opt/0000004516-1.tif', 
            '/mnt/project/data/grid_g05_low_v1/opt/0000034895-1.tif', 
            '/mnt/project/data/grid_g05_high_v1/opt/0002002383-1.tif'
        ])

        # ----------
        # Synthetic Data
        # ----------
        num_synthetic = 0

        for d in self.synthetic_dataset_names:
            train_list = self.data_dir / d / 'train.txt'
            
            with open(train_list, "r") as f:
                image_list = [ln.strip() for ln in f if ln.strip()]

            for image_name in image_list:
                img_filepath = self.data_dir / d / 'opt' / f'{image_name}.tif'

                if str(img_filepath) not in self.missing_files:
                    self.data.append({'image': img_filepath, 'source': self.source, 'target': self.target})
                    num_synthetic += 1

        print(f"# Synthetic: {num_synthetic}")

        # ----------
        # DFC18
        # ----------
        num_real = 0

        train_list = self.data_dir / "real" / "DFC18" / "DFC18" / "train.txt"

        with open(train_list, "r") as f:
            image_list = [ln.strip() for ln in f if ln.strip()]

        for image_name in image_list:
            img_filepath = self.data_dir / "real/DFC18/DFC18" / "opt" / f"{image_name}.tif"
            self.data.append({"image": img_filepath, "source": self.target, "target": self.source})
            num_real += 1

        # ----------
        # DFC19
        # ----------
        d = self.data_dir / "real" / "DFC19" / "opt"
        filenames = os.listdir(d)

        for filename in filenames:
            img_filepath = d / filename
            self.data.append({"image": img_filepath, "source": self.target, "target": self.source})
            num_real += 1

        # ----------
        # GeoNRW
        # ----------
        d = self.data_dir / "real" / "geonrw" / "data"
        d_dirs = [d_dir for d_dir in os.listdir(d) if os.path.isdir(d / d_dir)]

        for d_dir in d_dirs:
            filenames = [f for f in os.listdir(d / d_dir) if f.endswith(".jp2")]
            for filename in filenames:
                img_filepath = d / d_dir / filename
                self.data.append({"image": img_filepath, "source": self.target, "target": self.source})
                num_real += 1

        print(f"# Real: {num_real}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item = self.data[idx]
        path = str(item['image'])

        with Image.open(path) as im:
            im = im.convert("RGB")
            # -- Resize if not 512x512
            if im.size != (512, 512):
                im = im.resize((512, 512), resample=Image.BICUBIC)
            image = np.array(im)

        # -- Normalize to [-1,1]
        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=image, source=item['source'], txt=item['target'])
        

if __name__ == "__main__":
    ds = TrainDataset()
    print(len(ds))
    print(ds.data[:5])
                
