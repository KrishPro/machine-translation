"""
Written by KrishPro @ KP

filename: `train.py`
"""

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch

try:
    from model import Transformer
    from data import Dataset
except:
    from machine_translation.model import Transformer
    from machine_translation.data import Dataset


def train_step(i:int, src: torch.Tensor, tgt: torch.Tensor, model: Transformer, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, hparams: dict, device: torch.device, scaler: torch.cuda.amp.GradScaler, lr_scheduler: optim.lr_scheduler.LambdaLR) -> torch.Tensor:
    with torch.autocast(device_type=str(device)):
        tgt: torch.Tensor = tgt.to(device)

        out: torch.Tensor = model(src.to(device), tgt[:, :-1])

        loss: torch.Tensor = criterion(out.reshape(-1, hparams['dims']['tgt_vocab_size']), tgt[:, 1:].reshape(-1))

        scaler.scale(loss).backward()

        if i % hparams['accumulation_batch_size'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        return loss


def train(**kwargs):

    hparams = {
        'dims':  {
            'd_model': 128,
            'n_heads': 2,
            'dim_feedforward': 512,
            'n_layers': 3,
            'src_vocab_size': 30_000,
            'tgt_vocab_size': 30_000
        },
        'data_path': '.data/processed.txt',
        'accumulation_batch_size': 64,
        'batch_size': 1,
        'label_smoothing': 0.0, 
        'lr_factor': 100,
        'epochs': 10,
        'src': 'en',
        'tgt': 'fr',
        'overfit_one_batch': True,
        'warmup_steps': 4_000,
        'use_all_gpus': True
    }

    hparams.update(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(Transformer(**hparams['dims'], pad_idx=-1).to(device)) if hparams['use_all_gpus'] else Transformer(**hparams['dims'], pad_idx=-1).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=hparams['label_smoothing'])

    optimizer = optim.Adam(model.parameters(), lr=hparams['lr_factor'])
    
    lr_lambda = lambda step_num: (hparams['dims']['d_model'] ** -0.5) * min((step_num+1)**-0.5, (step_num+1)*(hparams['warmup_steps']**-1.5))
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler()

    dataset = Dataset(hparams['src'], hparams['tgt'], processed_path=hparams['data_path'])
    dataloader = data.DataLoader(dataset, batch_size=hparams['batch_size'], pin_memory=torch.cuda.is_available(), collate_fn=Dataset.collate_fn)

    if hparams['overfit_one_batch']:
        src, tgt = next(iter(dataloader))
        with tqdm(range(100_000)) as pbar:
            for i in pbar:
                loss = train_step(i, src, tgt, model, criterion, optimizer, hparams, device, scaler, lr_scheduler).detach()

                pbar.set_postfix(loss=loss.detach())

    for epoch in range(hparams['epochs']):

        with tqdm(dataloader, desc=f"EPOCH {epoch}") as pbar:

            for i, (src, tgt) in enumerate(pbar):
                loss = train_step(i, src, tgt, model, criterion, optimizer, hparams, device, scaler, lr_scheduler).detach()

                pbar.set_postfix(loss=loss.detach())

if __name__ == "__main__":
    train()