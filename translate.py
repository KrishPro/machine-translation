"""
Written by KrishPro @ KP

filename: `translate.py`
"""

from typing import List
from tokenizers import Tokenizer
import torch.nn.functional as F
from model import Transformer
import torch


class Translator:
    def __init__(self, ckpt_path: str, src_vocab_path: str, tgt_vocab_path: str, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        self.model = Transformer.from_ckpt(ckpt_path).to(device).eval()
  
        self.device = device
        self.mask_token = -1e+25
        self.max_len = 512

        self.src_vocab: Tokenizer = Tokenizer.from_file(src_vocab_path)
        self.tgt_vocab: Tokenizer = Tokenizer.from_file(tgt_vocab_path)

        self.sos_token, self.eos_token = self.src_vocab.token_to_id("[SOS]"), self.src_vocab.token_to_id("[EOS]")
        self.pad_token, self.unk_token = self.src_vocab.token_to_id("[PAD]"), self.src_vocab.token_to_id("[UNK]")

    def encode(self, src: torch.Tensor, attn_mask=None):
        """
        src.shape: (B, S, E)

        returns: (B, S, E)
        """

        for layer in self.model.encoder_layers:
            src = layer(src, attn_mask=attn_mask)

        return src

    def decode(self, src: torch.Tensor, tgt: torch.Tensor, attn_mask=None, memory_mask=None):
        """
        tgt.shape: (B, T, E)
        src.shape: (B, S, E)

        returns: (B, T, E)
        """

        tgt = self.model.pos_embedding(self.model.tgt_embedding(tgt))

        for layer in self.model.decoder_layers:
            tgt = layer(tgt, src, attn_mask=attn_mask, memory_mask=memory_mask)

        return F.softmax(self.model.output(tgt), dim=-1)


    @torch.no_grad()
    def translate(self, src_sentence: str, algorithm="greedy_decode", beam_width=None) -> List[str]:
        src = torch.tensor(self.src_vocab.encode(src_sentence).ids, device=self.device).unsqueeze(0)
    
        lm_mask = torch.empty(self.max_len, self.max_len, device=self.device).fill_(self.mask_token).triu_(1)
    
        src_encodings = self.model.pos_embedding(self.model.src_embedding(src))
        src_encodings = self.encode(src_encodings)

        if algorithm == "greedy_decode":
            tgt: torch.Tensor = torch.tensor([[self.sos_token]], device=self.device)

            i = 0
            while i < self.max_len:
                i += 1

                tgt_mask = lm_mask[:tgt.size(1), :tgt.size(1)].unsqueeze(0)

                pred: int = self.decode(src_encodings, tgt, tgt_mask)[:, -1, :].argmax(1).item()
                tgt = torch.cat([tgt, torch.tensor(pred, device=self.device).view(1, 1)], dim=1)

                if pred == self.eos_token:
                    break

            return zip([self.tgt_vocab.decode(tgt[0].tolist())], [1.])

        elif algorithm == "beam_search":
            assert beam_width > 0, "beam width should be greater than 0"

            # TODO: Improve the performance by using kv_cache for larger values of beam_width

            src_encodings = src_encodings.repeat(beam_width, 1, 1)
            tgt: torch.Tensor = torch.tensor([[self.sos_token] for _ in range(beam_width)], device=self.device)
            beams_conf = torch.tensor([1. for i in range(beam_width)], device=self.device)

            i = 0
            while i < self.max_len:
                i += 1

                tgt_mask = lm_mask[:tgt.size(1), :tgt.size(1)] 

                out: torch.Tensor = self.decode(src_encodings, tgt, tgt_mask)[:, -1, :]
      
                possiblities = torch.cat([tgt.repeat_interleave(out.size(1), dim=0), torch.arange(0, out.size(1), device=self.device).repeat(out.size(0)).unsqueeze(1)], dim=1)

                beams_conf, ids = torch.topk(out.reshape(-1) * beams_conf.repeat_interleave(out.size(1)), k=beam_width)

                tgt = possiblities[ids]

                if all([self.eos_token in t for t in tgt]):
                    break
            
            return zip(self.tgt_vocab.decode_batch(tgt.tolist()), beams_conf.softmax(0).tolist())

from timeit import default_timer
    

translator = Translator('/home/krish/Projects/machine-translation/.ignore/epoch=19-loss=3.400.ckpt', '.data/vocabs/vocab.en', '.data/vocabs/vocab.fr')

while True:
    english_sentence = input(">> ")
    start = default_timer()
    for i, (translation, conf) in enumerate(translator.translate(english_sentence, algorithm="beam_search", beam_width=64)):
        print(f"    {i+1}) ({conf*100:}%) {translation}")
        if (i+1) == 10:
            break

    print(f"Took {default_timer() - start}s")
    print()