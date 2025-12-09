from share import *

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from dataset import TrainDataset
from cycleNet.logger import ImageLogger, TextLogger
from cycleNet.model import create_model, load_state_dict

seed = 21
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Configs
model_name = 'real_long_prompt'
run_dir = Path('./runs') / model_name
log_dir = run_dir / 'logs'
ckpt_dir = run_dir / 'checkpoints'

resume_path = './models/cycle_sd21_ini.ckpt'
model_config = './models/cycle_v21.yaml'
batch_size_per_gpu = 4
gpus = 1
logger_freq = 50
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


def save_config():
    config_dir = run_dir / 'configs'
    os.makedirs(config_dir, exist_ok=True)

    # ----------
    # Save Model / Run Config
    # ----------
    config = OmegaConf.load(model_config)

    config["run"] = {
        "info": {
            "model_name": model_name,
            "run_dir": run_dir,
            "log_dir": log_dir,
            "ckpt_dir": ckpt_dir
        },
        "params": {
            "resume_path": resume_path,
            "learning_rate": learning_rate,
            "sd_locked": sd_locked,
            "only_mid_control": only_mid_control,
            "seed": seed
        }
    }

    OmegaConf.save(config, config_dir / 'config.yaml')


if __name__ == "__main__":
    # Misc
    print("Loading TrainDataset / DataLoader...")
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    print("Creating Trainer...")
    # logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    logger = TextLogger(log_every_n_steps=logger_freq)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=2000,
        save_last=False,
        monitor=None,
        auto_insert_metric_name=False,
        save_weights_only=True
    )

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=gpus, 
        precision=16, 
        callbacks=[checkpoint_cb, logger], 
        default_root_dir=log_dir, 
        max_steps=50000
    )
    print("Training CycleNet!")
    trainer.fit(model, dataloader)
