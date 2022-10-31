"""
Written by KrishPro @ KP

filename: `data.py`
"""


from typing import List, Tuple
import torch.utils.data as data
import pandas as pd
import torch

class Dataset(data.Dataset):
    def __init__(self, processed_path: str) -> None:
        super().__init__()
                
        self.data = pd.read_feather(processed_path)

    def __getitem__(self, idx: int):
        return tuple(self.data.iloc[idx])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data: List[Tuple[List[int], List[int]]]):
        src, tgt = zip(*data)

        src_max_len = max(map(len, src))
        tgt_max_len = max(map(len, tgt))

        src = [s.tolist() + ([0.] * (src_max_len-len(s))) for s in src]
        tgt = [t.tolist() + ([0.] * (tgt_max_len-len(t))) for t in tgt]

        return torch.tensor(src), torch.tensor(tgt)