import random
from typing import Iterable
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import register_plugin


def set_deterministic(seed=42):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@register_plugin("optim", "adamw")
def build_adamw(model, optim_cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["reg_w"]
    )


@register_plugin("engine", "train_engine")
def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    epoch: int,
):
    step = 0
    epoch_loss = 0

    model.train()
    for batch_data in data_loader:
        optimizer.zero_grad()
        batch_data = batch_data.cuda()
        #batch_loss, _, _ = model(batch_data, mask_ratio=cfg["MODEL"]["mask_ratio"])
        batch_loss, _pred, _mask = model(batch_data, mask_ratio=cfg["MODEL"]["mask_ratio"])
        #print(_mask.bool())
        #_pred[~(_mask.bool())] = 0.0

        #core = model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel,
        #                                  torch.nn.DataParallel)) else model
        #unp = core.unpatchify(_pred)


        
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        step += 1

        if step%100==1:
            _pred[~(_mask.bool())] = 0.0
            core = model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel,
                                          torch.nn.DataParallel)) else model
            unp = core.unpatchify(_pred)
            fig, ax = plt.subplots(1,2) 
            ax.flat[0].imshow(batch_data[0,0].detach().cpu())
            ax.flat[1].imshow(unp[0,0].detach().cpu())
            fig.savefig(f"{epoch}_{step}.png")
            print(f"{epoch}_{step}.png saved to disk")
            plt.close()

    return epoch_loss / step
