"""
Written by KrishPro @ KP

filename: `tmp.py`
"""

from model import Encoder, Decoder, PositionalEmbedding
import torch.nn.functional as F
import torch.nn as nn
import torch
import os


class EncoderModule(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, num_layers:int, src_vocab_size:int, dropout_p=0.1) -> None:
        super().__init__()
        self.max_len = 128
        self.pad_idx = 0
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, dropout_p=dropout_p, max_len=self.max_len)

        self.encoder_layers = nn.ModuleList([Encoder(d_model, n_heads, dim_feedforward, dropout_p=dropout_p) for _ in range(num_layers)])

    @classmethod
    def from_ckpt(cls, ckpt_path:str):
        ckpt = torch.load(ckpt_path)
        del ckpt['dims']['tgt_vocab_size']

        encoder = cls(**ckpt['dims'])

        own_modules = ['src_embedding', 'encoder_layers']
        state_dict = {k[7:]:v for k,v in ckpt['state_dict'].items() if any([k[7:].startswith(s) for s in own_modules])}

        encoder.load_state_dict(state_dict)
        return encoder.eval()

    def forward(self, src: torch.Tensor):
        # Removing padding
        if (src==self.pad_idx).nonzero().size(0) > 0:
            src = src[:, :(src==self.pad_idx).nonzero()[0,1]]

        src = self.pos_embedding(self.src_embedding(src))

        for layer in self.encoder_layers:
            src = layer(src)

        # Adding padding
        src = torch.cat([src, torch.zeros(1, self.max_len-src.size(1), src.size(2))], dim=1)

        return src

class DecoderModule(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, num_layers:int, tgt_vocab_size:int, dropout_p=0.1) -> None:
        super().__init__()
        self.mask_token = float('-inf')
        self.max_len = 128
        self.pad_idx = 0

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, dropout_p=0.1, max_len=self.max_len)

        self.decoder_layers = nn.ModuleList([Decoder(d_model, n_heads, dim_feedforward, dropout_p=dropout_p) for _ in range(num_layers)])
        self.lm_mask = torch.empty(self.max_len, self.max_len).fill_(self.mask_token).triu_(1)
        self.output = nn.Linear(d_model, tgt_vocab_size, bias=False)

    @classmethod
    def from_ckpt(cls, ckpt_path:str):
        ckpt = torch.load(ckpt_path)
        del ckpt['dims']['src_vocab_size']

        decoder = cls(**ckpt['dims'])

        own_modules = ['tgt_embedding', 'decoder_layers', 'output']
        state_dict = {k[7:]:v for k,v in ckpt['state_dict'].items() if any([k[7:].startswith(s) for s in own_modules])}

        decoder.load_state_dict(state_dict)
        return decoder.eval()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor):
        # Removing padding
        if (tgt==self.pad_idx).nonzero().size(0) > 0:
            tgt = tgt[:, :(tgt==self.pad_idx).nonzero()[0,1]]  

        if memory_pad_mask.nonzero().size(0) > 0:
            memory = memory[:, :memory_pad_mask.nonzero()[0, 1]]

        tgt = self.pos_embedding(self.tgt_embedding(tgt))

        tgt_mask = self.lm_mask[:tgt.size(1), :tgt.size(1)]


        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask)

        tgt = F.softmax(self.output(tgt[:, -1, :]), dim=1).argmax(1)

        return tgt
    

def export(ckpt_path: str, output_dir: str):
    encoder = EncoderModule.from_ckpt(ckpt_path)
    decoder = DecoderModule.from_ckpt(ckpt_path)

    src = torch.randint(0, 23, size=(1, 128))
    tgt = torch.randint(0, 23, size=(1, 128))

    with open(os.path.join(output_dir, 'encoder.onnx'), 'wb') as encoder_file, open(os.path.join(output_dir, 'decoder.onnx'), 'wb') as decoder_file:

        torch.onnx.export(encoder, args=(src,), f=encoder_file)
        torch.onnx.export(decoder, args=(tgt, encoder(src), src==decoder.pad_idx), f=decoder_file)
        
if __name__ == '__main__':
    # export('.ignore/Models/english-to-french.ckpt', '.ignore/Output/english-to-french')
    # export('.ignore/Models/english-to-german.ckpt', '.ignore/Output/english-to-german')
    # export('.ignore/Models/french-to-english.ckpt', '.ignore/Output/french-to-english')
    # export('.ignore/Models/german-to-english.ckpt', '.ignore/Output/german-to-english')
    pass