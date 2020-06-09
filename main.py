import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import load_config
from trainer import Trainer

from utils.vis import vis_preds, make_vis_inps

from rbdl.config.defaults import cfg
from rbdl.utils.model import init_model_weights
from rbdl.utils.setup_utils import setup_logger, setup_experiment_dir
from rbdl.utils.device_utils import get_device
from rbdl.data import build_dataloaders
from rbdl.train import build_loss_func
from rbdl.callbacks import ImageWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument("--comment", type=str, help="Comment on experiment")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args

def setup():
    args = parse_args()
    config = load_config(args)
    config = setup_experiment_dir(config)
    if args.seed: torch.manual_seed(args.seed) 
    shutil.copy(config.config_file, Path(config.experiment_dir) / "config.yaml")
    setup_logger(output = config.experiment_dir)
    return config

if __name__ == '__main__':
    config = setup()

    train_ds = SomeDataset(config) # write Dataset class or import from DL library
    val_ds = SomeDataset(config, train=False)
    dls = build_dataloaders(train_ds, val_ds, config=config, vis_fn=make_vis_inps()) # write vis_inps function

    device = get_device(use_gpu=True, device=1)
    model = SomeModel(config, device=device) # write Model class or import from DL library
    model = model.to(device)
    init_model_weights(config, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.opt.lr)
    loss_func = build_loss_func(config)

    trainer = Trainer(config, model, optimizer, loss_func, dls, device) # write loss_step for Trainer class
    trainer.train([ImageWriter(vis_fn=[vis_preds])]) # write vis_preds function