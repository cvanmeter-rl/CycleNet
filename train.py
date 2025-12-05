from share import *

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import TrainDataset
from real_and_synthethic_dataset import RealAndSynthethicTrainDataset
from cycleNet.logger import ImageLogger
from cycleNet.model import create_model, load_state_dict
import numpy as np
import torch
import argparse

torch.cuda.manual_seed(21)
np.random.seed(21)
torch.manual_seed(21)

# Configs
resume_path = './models/single_simple_prompt_Both_False_bs4_syn_and_real_data.ckpt'
log_path = './logs'
batch_size_per_gpu = 4
gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type",type=str,required=True,help="Name of dataset being used", default="synthethic_only")

    return parser.parse_args()


if __name__ == "__main__":

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cycle_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    if args.dataset == "synthethic_only":
        dataset = TrainDataset()
    elif args.dataset == "real_and_synthethic":
        dataset = RealAndSynthethicTrainDataset()
    else:
        return "Invalid Dataset Entered"
        
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    #logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    checkpoint_cb = ModelCheckpoint(
    dirpath=f"./checkpoints/models/single_simple_prompt_Both_False_bs4_syn_and_real_data/",
    filename="{step:06d}",
    save_top_k=-1,
    every_n_train_steps=5000,
    save_last=True,
    monitor=None,
    save_weights_only=True
    )
    trainer = pl.Trainer(accelerator="gpu", devices=gpus, precision=16, callbacks=[checkpoint_cb], default_root_dir=log_path,max_steps=50000)
    trainer.fit(model, dataloader)
