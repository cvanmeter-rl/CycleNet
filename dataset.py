import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.data_dir = Path('/mnt/synrs3d/SynRS3D/data/')
        source = 'synthetic satellite map render, flat colors, minimal texture, clean computer-generated style, no haze, no noise, no shadows'
        target = 'realistic highl-resolution satellite image, natural colors, rich textures, realistic lighting, sensor noise, atmospheric haze, soft shadows'
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
            train_list = self.data_dir / d / 'train.txt'
            with open(train_list, "r") as f:
                image_list = [ln.strip() for ln in f if ln.strip()]
            for image_name in image_list:
                img_filepath = self.data_dir / d / 'opt' / f'{image_name}.tif'
                self.data.append({'image':img_filepath, source: self.source, target: self.target})

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item = self.data[idx]

        image = cv2.imread(item['image'])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #normalize to -1,1
        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=image, source=source, target=target)
        

if __name__ == "__main__":
    ds = TrainDataset()
    print(len(ds))
    print(ds.data[:5])
                
