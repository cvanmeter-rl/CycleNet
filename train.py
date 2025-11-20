from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import TrainDataset
from cycleNet.logger import ImageLogger
from cycleNet.model import create_model, load_state_dict

# Configs
resume_path = './models/cycle_sd21_single_simple_prompt_frozenSD_allControl.ckpt'
log_path = './logs'
batch_size_per_gpu = 1
gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


if __name__ == "__main__":

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cycle_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    trainer = pl.Trainer(accelerator="gpu", devices=gpus, precision=16, callbacks=[], default_root_dir=log_path)
    trainer.fit(model, dataloader)
