import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from test_dataset import TestDataset

from cycleNet.model import create_model, load_state_dict

import argparse

BATCH_SIZE = 4
CONFIG_PATH = "./models/cycle_v21.yaml"
ckpt_path = Path('/mnt/cyclenet/CycleNet/checkpoints/models/single_simple_prompt_Both_False_bs4_syn_and_real_data/step=044999-v1.ckpt')
#output_path = Path(f'/mnt/cyclenet/CycleNet/full_test/{ckpt_path.parent.name}')
output_path = Path(f'/mnt/cyclenet/CycleNet/full_test/longer_prompt_Both_False_bs4_syn_and_real_data')

def main():
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(output_path, exist_ok=True)

  dataset = TestDataset()

  dataloader = DataLoader(dataset, batch_size=4,shuffle=False,num_workers=0)

  
  for d in dataset.synthetic_dataset_names:
    p = output_path / d / 'opt' 
    os.makedirs(p,exist_ok=True)
    print(f'made {p} directory')

  print('loading model')
  model = create_model(CONFIG_PATH).to(device)
  state = load_state_dict(ckpt_path, location='cpu')
  model.load_state_dict(state)
  model.eval()
  print('model loaded')

  cfg = 1.00
  with torch.no_grad():
    for idx, batch in enumerate(dataloader):
      paths = batch['img_path']
    
      for k, v in batch.items():
        if isinstance(v, torch.Tensor):
          batch[k] = v.to(device)
          
      model_batch = {k:v for k,v in batch.items() if k != 'img_path'}
      
      if cfg == 1.00:
        logs = model.log_images(model_batch, split="test", unconditional_guidance_scale=cfg, sample=True)
        key = 'samples'
      else:
        logs = model.log_images(model_batch, split="test", unconditional_guidance_scale=cfg)
        key = f'samples_cfg_scale_{cfg:.2f}'
        
      x = logs[key]
      x = x.float().clamp(-1.0, 1.0)
      x = (x + 1.0) / 2.0

      for i in range(x.size(0)):
        p = Path(paths[i])
        dataset_name = p.parent.parent.name
        image_name = p.stem
        output = output_path / dataset_name / 'opt' / f'{image_name}.tif'
        save_image(x[i].cpu(),output)
        print(f'Saved at {output}')

  print('Done')

if __name__ == "__main__":
    main() 
        
      
  

  

  
  
