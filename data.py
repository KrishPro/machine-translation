"""
Written by KrishPro @ KP

filename: `data.py`
"""


from typing import List, Tuple
import torch.utils.data as data
from process_data import count_lines
import torch

class Dataset(data.Dataset):
    def __init__(self, processed_path: str, chunk_size:int = 2**18) -> None:
        super().__init__()
        
        self.processed_file = open(processed_path)
        self.len = count_lines(processed_path)
        self.chunk_size: int = chunk_size

        self._reset()

    def _load_chunk(self):
        if not hasattr(self, 'previous_left'): self.previous_left = ''

        chunk = (self.previous_left + self.processed_file.read(self.chunk_size)).splitlines()

        self.previous_left = chunk[-1]

        chunk = filter(lambda x:x!='', chunk[:-1])

        chunk = list(map(lambda x: list(map(lambda x: list(map(int, x.split(' '))), x.split('\t'))), chunk))

        return chunk

    def _reset(self):
        self.processed_file.seek(0)
        self.current_chunk_start = 0
        self.current_chunk = self._load_chunk()
        self.current_chunk_end = len(self.current_chunk) - 1


    def load_chunk(self):
        self.current_chunk_start += len(self.current_chunk)
  
        self.current_chunk = self._load_chunk()
 
        self.current_chunk_end += len(self.current_chunk)

    def __getitem__(self, idx: int):
        if idx > self.current_chunk_end:
            self.load_chunk()

        try:
            return self.current_chunk[idx - self.current_chunk_start]
        except IndexError:
            self._reset()
            raise IndexError

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data: List[Tuple[List[int], List[int]]]):
        src, tgt = zip(*data)

        src = [s + ([0.] * (max(map(len, src)) - len(s))) for s in src]
        tgt = [t + ([0.] * (max(map(len, tgt)) - len(t))) for t in tgt]

        return torch.tensor(src), torch.tensor(tgt)
