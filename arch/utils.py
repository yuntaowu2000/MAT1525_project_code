import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_eval(model: torch.nn.Module, dataloader):
    model.eval() # switch to eval model, will turn off randomness like dropout
    eval_losses = defaultdict(float)
    num_batches = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval")):
        batch = batch[0].to(model.device)
        losses = model.test_step(batch)
        for k, v in losses.items():
            eval_losses[k] += v.item()
        num_batches += 1

    for k, v in eval_losses.items():
        eval_losses[k] = v / num_batches

    return eval_losses


def save_loss_to_csv(train_loss_dict: dict[str, float], 
                     test_loss_dict: dict[str, float], 
                     epoch: int, csv_file: str):
    record = {"epoch": epoch}
    record = record | {f"train_{k}": v for k, v in train_loss_dict.items()}
    record = record | {f"test_{k}": v for k, v in test_loss_dict.items()}
    df = pd.DataFrame([record])
    df.to_csv(csv_file, mode="a+", index=False, header=not os.path.exists(csv_file))