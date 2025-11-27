import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.utilities.distributed import rank_zero_only


# class TextLogger(Callback):
#     def __init__(self, log_every_n_steps=50):
#         self.log_every_n_steps = log_every_n_steps

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if trainer.global_step % self.log_every_n_steps == 0:
#             # Try to access the loss from the outputs or the logged metrics
#             # Note: outputs is usually a dict containing 'loss' if training_step returns it
#             loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            
#             # If not in outputs, try trainer.callback_metrics (logged via self.log)
#             if loss is None:
#                 loss = trainer.callback_metrics.get("train_loss") or trainer.callback_metrics.get("loss")
            
#             # Format loss to 4 decimal places if available
#             loss_val = f"{loss:.4f}" if loss is not None else "N/A"
            
#             current_lr = trainer.optimizers[0].param_groups[0]['lr']
            
#             print(
#                 f"Epoch {trainer.current_epoch} | "
#                 f"Step {trainer.global_step} | "
#                 f"Loss: {loss_val} | "
#                 f"LR: {current_lr:.2e}"
#             )


class ProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.file = sys.stdout # Force writing to standard output
        return bar
    
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.file = sys.stdout # Force writing to standard output
        return bar
    

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, every_n_train_steps=1000, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
