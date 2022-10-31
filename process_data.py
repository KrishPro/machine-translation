"""
Written by KrishPro @ KP

filename: `process_data.py`
"""

import os
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import List, Tuple
from unidecode import unidecode
import pandas as pd
import re
from vocab import create_vocab

class CleanText:
    html_cleaner = re.compile('<.*?>') 
    url_remover = re.compile(r'https?://\S+')

    @classmethod
    def remove_html_tags(cls, text: str):
        text: str = re.sub(cls.html_cleaner, '', text)
        return text

    @classmethod
    def remove_urls(cls, text: str):
        text: str = re.sub(cls.url_remover, '', text)
        return text

    @classmethod
    def filter_text(cls, text: str):
        return not text.isdigit()

    @classmethod
    def c(cls, text: str):
        text: str = cls.remove_html_tags(text)
        text: str = cls.remove_urls(text)
        text: str = unidecode(text)
        text: str = text if cls.filter_text(text) else ''
        text: str = text.replace("\n", " ")
        return text

def count_lines(input_path:str):
    with open(input_path) as input_file:
        num_sentences = 0
        for _ in input_file:
            num_sentences += 1
    return num_sentences

def load_raw_data(src_input_path:str, tgt_input_path:str):
    # Getting the number of sentences
    src_num_sentences, tgt_num_sentences =  count_lines(src_input_path), count_lines(tgt_input_path)
    assert src_num_sentences == tgt_num_sentences, "Both files should have same number of sentences"
    num_sentences = src_num_sentences

    with open(src_input_path) as src_input_file, open(tgt_input_path) as tgt_input_file:
        for src, tgt in tqdm(zip(src_input_file, tgt_input_file), total=num_sentences, desc="Loading and processing sentences"):
            src, tgt = CleanText.c(src.strip()), CleanText.c(tgt.strip())
            if src and tgt:
                yield src, tgt


def process_raw_data(src_path: str, tgt_path: str, src_out_path: str, tgt_out_path: str):
    data: List[Tuple[str, str]] = list(load_raw_data(src_path, tgt_path))

    print("Dumping sentences...")
    src, tgt = zip(*data)

    with open(src_out_path, 'w') as src_out_file:
        src_out_file.write('\n'.join(src))

    with open(tgt_out_path, 'w') as tgt_out_file:
        tgt_out_file.write('\n'.join(tgt))

def load_data(src_input_path: str, tgt_input_path: str):
    src_num_sentences, tgt_num_sentences =  count_lines(src_input_path), count_lines(tgt_input_path)
    assert src_num_sentences == tgt_num_sentences, "Both files should have same number of sentences"
    num_sentences = src_num_sentences

    with open(src_input_path) as src_input_file, open(tgt_input_path) as tgt_input_file:
        for src, tgt in tqdm(zip(src_input_file, tgt_input_file), total=num_sentences, desc="Loading and tokenizing sentences"):
            yield src, tgt

def process_data(src_input_path: str, tgt_input_path: str, src_vocab_path: str, tgt_vocab_path: str, output_path: str):
    data: List[Tuple[str, str]] = list(load_data(src_input_path, tgt_input_path))

    src_tokenizer: Tokenizer = Tokenizer.from_file(src_vocab_path)
    tgt_tokenizer: Tokenizer = Tokenizer.from_file(tgt_vocab_path)

    print("Dumping sentences...")
    src, tgt = zip(*data)

    src_tokens: List[List[int]] = [encoding.ids for encoding in src_tokenizer.encode_batch(src)]
    tgt_tokens: List[List[int]] = [encoding.ids for encoding in tgt_tokenizer.encode_batch(tgt)]

    dataframe = pd.DataFrame({'src': src_tokens, 'tgt': tgt_tokens})

    dataframe.to_feather(output_path)



if __name__ == '__main__':
    process_raw_data('.data/raw/train.tags.fr-en.en', '.data/raw/train.tags.fr-en.fr', '.data/sentences.en', '.data/sentences.fr')
    print()

    print("Creating vocabs")
    if not os.path.exists('.data/vocabs'): os.makedirs('.data/vocabs')
    create_vocab(['.data/sentences.en'], '.data/vocabs/vocab.en')
    create_vocab(['.data/sentences.fr'], '.data/vocabs/vocab.fr')
    print()

    process_data('.data/sentences.en', '.data/sentences.fr', '.data/vocabs/vocab.en', '.data/vocabs/vocab.fr', '.data/processed.dt')