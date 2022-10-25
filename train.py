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


def train(**kwargs):

    hparams = {
        'dims':  {
            'd_model': 512,
            'n_heads': 8,
            'dim_feedforward': 2048,
            'n_layers': 4,
            'src_vocab_size': 30_000,
            'tgt_vocab_size': 30_000
        },
        'data_path': '.data/processed.txt',
        'accumulation_batch_size': 2,
        'batch_size': 1,
        'label_smoothing': 0.1, 
        'learning_rate': 3e-4,
        'epochs': 10,
        'src': 'en',
        'tgt': 'fr'
    }

    hparams.update(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(**hparams['dims'], pad_idx=-1).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=hparams['label_smoothing'])

    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    dataset = Dataset(hparams['src'], hparams['tgt'], processed_path=hparams['data_path'])
    dataloader = data.DataLoader(dataset, batch_size=hparams['batch_size'], pin_memory=torch.cuda.is_available(), collate_fn=Dataset.collate_fn)

    for epoch in range(hparams['epochs']):

        with tqdm(dataloader, desc=f"EPOCH {epoch}") as pbar:

            for i, (src, tgt) in enumerate(pbar):
                tgt: torch.Tensor = tgt.to(device)

                out: torch.Tensor = model(src.to(device), tgt[:, :-1])

                loss: torch.Tensor = criterion(out.reshape(-1, hparams['dims']['tgt_vocab_size']), tgt[:, 1:].reshape(-1))

                loss.backward()

                if i % hparams['accumulation_batch_size'] == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(loss=loss.item())

train()