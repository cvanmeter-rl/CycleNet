from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path

class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.data_dir = Path('/mnt/synrs3d/SynRS3D/data/')
        #self.source = 'synthetic satellite map render, flat colors, minimal texture, clean computer-generated style, no haze, no noise, no shadows'
        #self.target = 'realistic high-resolution satellite image, natural colors, rich textures, realistic lighting, sensor noise, atmospheric haze, soft shadows'
        self.source = 'synthetic satellite image'
        self.target = 'real satellite image'
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
        missing = 0
        for d in self.synthetic_dataset_names:
            train_list = self.data_dir / d / 'train.txt'
            with open(train_list, "r") as f:
                image_list = [ln.strip() for ln in f if ln.strip()]
            for image_name in image_list:
                img_filepath = self.data_dir / d / 'opt' / f'{image_name}.tif'
                if Path(img_filepath).exists():
                    self.data.append({'image':img_filepath, 'source': self.source, 'target': self.target})
                else:
                    missing += 1
        print(f"[INFO] Loaded {len(self.data)} images, skipped {missing} missing files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item = self.data[idx]
        path = str(item['image'])

        with Image.open(path) as im:
            im = im.convert("RGB")
            image = np.array(im)
        #normalize to -1,1
        image = (image.astype(np.float32) / 127.5) - 1.0
        #HWC -> CHW
        #image = torch.from_numpy(image).permute(2,0,1)

        return dict(jpg=image, source=item['source'], txt=item['target']), path
        

if __name__ == "__main__":
    ds = TrainDataset()
    print(len(ds))
    print(ds.data[:5])
                
