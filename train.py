"""
Written by KrishPro @ KP

filename: `train.py`
"""

import os
try:
    from model import Transformer
    from data import Dataset
except:
    from machine_translation.model import Transformer
    from machine_translation.data import Dataset


import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
import time

def train_step(model: Transformer, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, src: torch.Tensor, tgt: torch.Tensor, optimizer_step=True, val_step=False):

    out: torch.Tensor = model(src, tgt[:, :-1])
    # out.shape: (B, T-1, V)

    loss: torch.Tensor = criterion(out.reshape(-1, hparams['dims']['tgt_vocab_size']), tgt[:, 1:].reshape(-1))

    if val_step:
        return loss.detach()

    loss.backward()

    if optimizer_step:
        optimizer.step()
        optimizer.zero_grad()

    return loss.detach()

hparams = {
    "dims": {
        "d_model": 128,
        "n_heads": 2,
        "dim_feedforward": 512,
        "num_layers": 2,
        "src_vocab_size": 30_000,
        "tgt_vocab_size": 30_000
    },

    "data_path": '.data/processed.dt',
    "learning_rate": 1e-3,
    "batch_size": 4,
    "accumulative_batch_size": 4,
    "overfit_one_batch": {"epochs": 1_000, "batch_size": 4},
    "epochs": 10,
    "log_interval": 50,
    "test_ratio": 0.1,
    "ckpt_dir": '.ignore',
    "dropout": 0.1,
    "smooth_logs": True,
    "label_smoothing": 0.1,
}

def train(hparams = hparams, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = nn.DataParallel(Transformer(**hparams['dims'], dropout_p=hparams['dropout']).to(device))

    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=hparams['label_smoothing'])

    dataset = Dataset(hparams['data_path'])
    test_size = int(len(dataset) * hparams['test_ratio'])

    train_dataset, test_dataset = data.random_split(dataset, [len(dataset)-test_size, test_size])

    train_dataloader = data.DataLoader(train_dataset, hparams['batch_size'], num_workers=os.cpu_count(), pin_memory=device == "cuda", collate_fn=Dataset.collate_fn)
    test_dataloader = data.DataLoader(test_dataset, hparams['batch_size'], num_workers=os.cpu_count(), pin_memory=device == "cuda", collate_fn=Dataset.collate_fn)

    if hparams["overfit_one_batch"]:
        print("Overfitting single batch")
        src, tgt = next(iter(train_dataloader))

        src: torch.Tensor = src.to(device).long()[:hparams['overfit_one_batch']['batch_size']]
        tgt: torch.Tensor = tgt.to(device).long()[:hparams['overfit_one_batch']['batch_size']]

        prev_frame_time = 0
        total_epochs = hparams['overfit_one_batch']['epochs']
        for i in range(total_epochs):

            # Calculating fps
            new_frame_time = time.time()
            prev_frame_time = time.time() if prev_frame_time == 0 else prev_frame_time
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            # Calculating eta
            eta = (total_epochs - i) / fps
            mm, ss = divmod(eta, 60)
            hh, mm = divmod(mm, 60)

            loss = train_step(model, optimizer, criterion, src, tgt)

            log = f"{i:05d}/{total_epochs} | ({i/total_epochs*100:.3f}%) | [{fps:.3f}it/s] | eta: {int(hh):02d}:{int(mm):02d}:{int(ss):02d} | Loss: {loss:.5f}"

            if i != 0: print(log, end='\r' if hparams['smooth_logs'] else '\n')
        print(log)
        print("\n")
        
    print("Starting training")
    prev_frame_time = 0
    total_datapoints = (len(train_dataloader.dataset) + len(test_dataloader.dataset)) / hparams["batch_size"]

    train_loss_history = []
    test_loss_history = []

    try:
        for epoch in range(hparams['epochs']):
            for i, (src, tgt) in enumerate(train_dataloader):

                src: torch.Tensor = src.to(device).long()
                tgt: torch.Tensor = tgt.to(device).long()

                optimizer_step = i % hparams['accumulative_batch_size'] == 0

                loss = train_step(model, optimizer, criterion, src, tgt, optimizer_step=optimizer_step)
                train_loss_history.append(loss)

                # Calculating fps
                new_frame_time = time.time()
                prev_frame_time = time.time() if prev_frame_time == 0 else prev_frame_time
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # Calculating eta
                eta = (total_datapoints - i) / fps
                mm, ss = divmod(eta, 60)
                hh, mm = divmod(mm, 60)

                log = f"epoch {epoch:02d}/{hparams['epochs']} | {i:06d}/{total_datapoints} | ({i/total_datapoints*100:.3f}%) | [{fps:.3f}it/s] | eta: {int(hh):02d}:{int(mm):02d}:{int(ss):02d} | Loss: {loss:.5f}"

                if i % hparams['log_interval'] == 0 and i != 0: print(log, end='\r' if hparams['smooth_logs'] else '\n')

            train_i = i
            

            with torch.no_grad():
                model.eval()
                for i, (src, tgt) in enumerate(test_dataloader):
                    i += train_i

                    src: torch.Tensor = src.to(device).long()
                    tgt: torch.Tensor = tgt.to(device).long()

                    optimizer_step = i % hparams['accumulative_batch_size'] == 0

                    loss = train_step(model, optimizer, criterion, src, tgt, optimizer_step=optimizer_step, val_step=True)
                    test_loss_history.append(loss)

                    # Calculating fps
                    new_frame_time = time.time()
                    prev_frame_time = time.time() if prev_frame_time == 0 else prev_frame_time
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time

                    # Calculating eta
                    eta = (total_datapoints - i) / fps
                    mm, ss = divmod(eta, 60)
                    hh, mm = divmod(mm, 60)

                    log = f"[VAL] epoch {epoch:02d}/{hparams['epochs']} | {i:06d}/{total_datapoints} | ({i/total_datapoints*100:.3f}%) | [{fps:.3f}it/s] | eta: {int(hh):02d}:{int(mm):02d}:{int(ss):02d} | Loss: {loss:.5f}"

                    if i % hparams['log_interval'] == 0 and i != 0: print(log, end='\r' if hparams['smooth_logs'] else '\n')

                log = f"[VAL] epoch {epoch:02d}/{hparams['epochs']} | {i:06d}/{total_datapoints} | ({i/total_datapoints*100:.3f}%) | [{fps:.3f}it/s] | eta: {int(hh):02d}:{int(mm):02d}:{int(ss):02d} | Loss: {sum(test_loss_history[-(i-train_i):])/(i-train_i):.5f}"
                model.train()
            print(log)
            torch.save({'state_dict': model.state_dict(), 'hparams': hparams, 'dims': hparams['dims'], 'history': {'train': train_loss_history, 'test': test_loss_history}}, os.path.join(hparams['ckpt_dir'], f"epoch={epoch}-loss={loss:.3f}.ckpt"))
    except Exception as e:
        print(f"Error: {e}")

    finally:
        print(log)


if __name__ == '__main__':
    train()