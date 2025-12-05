from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import TrainDataset
from cycleNet.logger import ImageLogger, TextLogger
from cycleNet.model import create_model, load_state_dict

torch.cuda.manual_seed(21)
np.random.seed(21)
torch.manual_seed(21)

# Configs
resume_path = './models/cycle_sd21_ini.ckpt'
model_name = 'simple_prompt'
log_path = f'./logs/{model_name}'
batch_size_per_gpu = 4
gpus = 1
logger_freq = 50
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


if __name__ == "__main__":

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cycle_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    print("Loading TrainDataset / DataLoader...")
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    print("Creating Trainer...")
    # logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/{model_name}/",
        filename="step-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=2000,
        save_last=True,
        monitor=None,
        auto_insert_metric_name=False,
        save_weights_only=True
    )
    logger = TextLogger(log_every_n_steps=logger_freq)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=gpus, 
        precision=16, 
        callbacks=[checkpoint_cb, logger], 
        default_root_dir=log_path, 
        max_steps=50000
    )
    print("Training CycleNet!")
    trainer.fit(model, dataloader)
