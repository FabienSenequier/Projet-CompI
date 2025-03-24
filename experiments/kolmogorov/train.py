#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *

# Force output into your Google Drive experiment folder
PATH = Path("/content/drive/MyDrive/FISE 3A/UE_F_Computational Imaging/Project/sda-master/experiments/kolmogorov")

# CONFIG = {
#     # Architecture
#     # 'window': 5,
#     'window': 3,
#     # 'embedding': 64,
#     'embedding': 32,
#     'hidden_channels': (96, 192, 384),
#     'hidden_blocks': (3, 3, 3),
#     'kernel_size': 3,
#     'activation': 'SiLU',
#     # Training
#     # 'epochs': 4096,
#     'epochs': 200,
#     # 'batch_size': 32,
#     'batch_size': 4,
#     'optimizer': 'AdamW',
#     'learning_rate': 2e-4,
#     'weight_decay': 1e-3,
#     'scheduler': 'linear',
# }

CONFIG = {
    'window': 3,
    'embedding': 16,
    'hidden_channels': (32, 64, 128),
    'hidden_blocks': (2, 2, 2),
    'kernel_size': 3,
    'activation': 'SiLU',
    'epochs': 100,
    'batch_size': 2,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}

@job(array=3, cpus=4, gpus=1, ram='16GB', time='24:00:00')
def train(i: int):
    run = wandb.init(project='sda-kolmogorov', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    window = CONFIG['window']
    score = make_score(**CONFIG)
    # sde = VPSDE(score.kernel, shape=(window * 2, 64, 64)).cuda()
    sde = VPSDE(score.kernel, shape=(window * 2, 64, 64))

    # Data
    trainset = TrajectoryDataset(PATH / 'data/train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(PATH / 'data/valid.h5', window=window, flatten=True)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        # device='cuda',
        device='cpu',
        **CONFIG,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation
    x = sde.sample((2,), steps=64).cpu()
    x = x.unflatten(1, (-1, 2))
    w = KolmogorovFlow.vorticity(x)

    run.log({'samples': wandb.Image(draw(w))})
    run.finish()


# if __name__ == '__main__':
    # schedule(
    #     train,
    #     name='Training',
    #     backend='slurm',
    #     export='ALL',
    #     env=['export WANDB_SILENT=true'],
    # )

if __name__ == '__main__':
    train(0)  # You can use 0, 1, or 2 for the job index (originally an array job)
