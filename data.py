"""
Written by KrishPro @ KP

filename: `data.py`
"""

import os
import torch
from tqdm import tqdm
import torch.utils.data as data
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence

def read_sentences(file_path: str):
    with open(file_path) as file:
        sentences = file.read().split("\n")
    return sentences

def tokenize_data(src_path:str, src_vocab_path:str, tgt_path: str, tgt_vocab_path:str, output_path: str):
    '''This function do all the pre-processing needed, so that on run-time we can be as fast as possible'''
    
    src_name: str = os.path.splitext(src_path)[-1].strip('.')
    tgt_name: str = os.path.splitext(tgt_path)[-1].strip('.')

    src_tokenizer: Tokenizer = Tokenizer.from_file(src_vocab_path)
    tgt_tokenizer: Tokenizer = Tokenizer.from_file(tgt_vocab_path)

    
    with open(output_path, 'w') as output_file:

        src_sentences = list(filter(lambda x: x.strip()!="", read_sentences(src_path)))
        tgt_sentences = list(filter(lambda x: x.strip()!="", read_sentences(tgt_path)))

        src_tokenized = list(map(lambda x: " ".join(map(str, x.ids)), src_tokenizer.encode_batch(src_sentences)))
        tgt_tokenized = list(map(lambda x: " ".join(map(str, x.ids)), tgt_tokenizer.encode_batch(tgt_sentences)))

        output_file.write(f"{src_name}\t{tgt_name}\n")
        for src_tokens, tgt_tokens in tqdm(zip(src_tokenized, tgt_tokenized), total=len(src_tokenized)):
            
            output_file.write(f"{src_tokens}\t{tgt_tokens}\n")

class Dataset(data.Dataset):
    """
    Use Like:
    dataset = Dataset('en', 'fr', processed_path='.data/processed.txt')
    dataloader = data.DataLoader(dataset, batch_size=B, pin_memory=True, collate_fn=Dataset.collate_fn)
    """
    def __init__(self, src_name:str, tgt_name:str, processed_path:str) -> None:
        super().__init__()

        self.src_name = src_name
        self.tgt_name = tgt_name

        self.data = self._load_data(processed_path)

    def _load_data(self, processed_path:str):
        with open(processed_path) as processed_file:
            processed_src_name, processed_tgt_name = next(processed_file).strip().split('\t')

            assert sorted((self.src_name, self.tgt_name)) == sorted((processed_src_name, processed_tgt_name))

            src_index = (processed_src_name, processed_tgt_name).index(self.src_name)
            tgt_index = (processed_src_name, processed_tgt_name).index(self.tgt_name)

            data = []

            for processed_sentence in processed_file:
                processed_sentence = list(map(lambda x: list(map(int, x.split(' '))), processed_sentence.strip().split('\t')))
                data.append((processed_sentence[src_index], processed_sentence[tgt_index]))

            return data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate_fn(data):
        src, tgt = zip(*data)

        src = pad_sequence([torch.tensor(s) for s in src])
        tgt = pad_sequence([torch.tensor(t) for t in tgt])

        return src.transpose(0, 1), tgt.transpose(0, 1)