"""
Written by KrishPro @ KP

filename: `model.py`
"""

import math
import torch
import numpy as np
import torch.nn as nn
from typing import Iterable
import torch.nn.functional as F


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None):
    """
    Q.shape: (B, QS, E/H)
    K.shape: (B, KS, E/H)
    V.shape: (B, KS, E/H)
    attn_mask.shape: (B*H, QS, KS)

    returns: (B, QS, E/H)

    Note: QS will be equal to KS, in simple attention layers
    """

    attn_mask = attn_mask if attn_mask is not None else torch.zeros(Q.size(1), K.size(1), device=Q.device)
    
    return torch.bmm(F.softmax(torch.baddbmm(attn_mask, Q, K.transpose(-2, -1))/math.sqrt(K.size(-1)), dim=-1), V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int) -> None:
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model, bias=False) # idk, why bias is only not in WK (i just copied openai/whisper) ?
        self.WV = nn.Linear(d_model, d_model)

        self.WO = nn.Linear(d_model, d_model)


    def forward(self, x: torch.Tensor, xa: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, QS, E)
        xa.shape: (B, KS, E)
        attn_mask: (B, QS, KS)

        returns: (B, QS, E)

        Note: QS will be equal to KS, in simple attention layers
        """

        B, QS, _ = x.shape

        KS = QS if xa is None else xa.size(1)
        
        assert xa is None or (xa.size(0) == x.size(0)), "Batchsize should be same"

        assert ((xa is None) and x.size(2) == self.d_model) or (x.size(2) == xa.size(2) == self.d_model), "Embed_dims should be same for x and xa and be equal to d_model"

        Q: torch.Tensor = self.WQ(x)
        K: torch.Tensor = self.WK(x if xa is None else xa)
        V: torch.Tensor = self.WV(x if xa is None else xa)

        assert Q.size(0) == K.size(0) == V.size(0), "Batchsize should be same"
        assert K.size(1) == V.size(1), "Seqlen of K and V should be same"
        assert Q.size(2) == K.size(2) == V.size(2) == self.d_model, "embed_dims should be same for Q,K,V and be equal to self.d_model"

        Q = Q.view(B, QS, self.n_heads, self.d_head).permute(0, 2, 1, 3).reshape(B*self.n_heads, QS, self.d_head)
        K = K.view(B, KS, self.n_heads, self.d_head).permute(0, 2, 1, 3).reshape(B*self.n_heads, KS, self.d_head)
        V = V.view(B, KS, self.n_heads, self.d_head).permute(0, 2, 1, 3).reshape(B*self.n_heads, KS, self.d_head)

        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, QS, KS).reshape(B*self.n_heads, QS, KS)
        # attn_mask.shape: (B*H, QS, KS)

        output = self_attention(Q, K, V, attn_mask=attn_mask)
        # output.shape: (B*H, QS, E/H)

        output = self.WO(output.view(B, self.n_heads, QS, self.d_head).permute(0, 2, 1, 3).reshape(B, QS, self.n_heads*self.d_head))

        return output


class Encoder(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float) -> None:
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        src.shape      : (B, QS, E)
        src_mask.shape : (QS, QS)

        returns: (B, QS, E)
        """

        src = self.norm1(src + self.dropout(self.self_attn(src, attn_mask=src_mask)))

        src = self.norm2(src + self.dropout(self.feedforward(src)))

        return src


class Decoder(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float) -> None:
        super().__init__()
        
        self.self_attn1 = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn2 = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tgt: torch.Tensor,  memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor):
        """
        tgt.shape         : (B, QS, E)
        memory.shape      : (B, KS, E)
        tgt_mask.shape    : (QS, QS)
        memory_mask.shape : (QS, KS)

        returns: (B, QS, E)
        """

        tgt = self.norm1(tgt + self.dropout(self.self_attn1(tgt, attn_mask=tgt_mask)))

        tgt = self.norm2(tgt + self.dropout(self.self_attn2(tgt, memory, attn_mask=memory_mask)))

        tgt = self.norm3(tgt + self.dropout(self.feedforward(tgt)))

        return tgt



class Transformer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, n_layers:int, src_vocab_size:int, tgt_vocab_size:int, dropout_p:float=0.1, pad_idx:int=0) -> None:
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embeddings = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embeddings = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_embeddings = nn.Parameter(self.generate_sinusoids(4_000, d_model))

        self.encoder: Iterable[Encoder] = nn.ModuleList([Encoder(d_model, n_heads, dim_feedforward, dropout_p) for _ in range(n_layers)])
        
        self.decoder: Iterable[Decoder] = nn.ModuleList([Decoder(d_model, n_heads, dim_feedforward, dropout_p) for _ in range(n_layers)])

    @staticmethod
    def generate_sinusoids(length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


    @staticmethod
    def generate_tgt_mask(T:int):
        return torch.empty(T, T).fill_(-np.inf).triu_(1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        src.shape: (B, S)
        tgt.shape: (B, T)

        src_mask.shape: (S, S)
        tgt_mask.shape: (T, T)
        memory_mask.shape: (T, S)
        """

        assert src.device == tgt.device, "Both SRC & TGT should be on the same device"

        assert src.size(0) == tgt.size(0), "Batch-size should be same"

        B, S = src.shape
        _, T = tgt.shape

        if not src_mask: src_mask = torch.zeros(src.size(1), src.size(1), device=src.device)
        if not tgt_mask: tgt_mask = torch.zeros(tgt.size(1), tgt.size(1), device=src.device)
        if not memory_mask: memory_mask = torch.zeros(tgt.size(1), src.size(1), device=tgt.device)

        src_pad_mask = torch.zeros(B, S, device=src.device).masked_fill(src == self.pad_idx, -np.inf)
        tgt_pad_mask = torch.zeros(B, T, device=tgt.device).masked_fill(tgt == self.pad_idx, -np.inf)

        src_mask = src_mask        + src_pad_mask.unsqueeze(2) + src_pad_mask.unsqueeze(1)
        tgt_mask = tgt_mask        + tgt_pad_mask.unsqueeze(2) + tgt_pad_mask.unsqueeze(1)
        memory_mask = memory_mask  + tgt_pad_mask.unsqueeze(2) + src_pad_mask.unsqueeze(1)

        # src_mask.shape: (B, S, S)
        # tgt_mask.shape: (B, T, T)
        # memory_mask.shape: (B, T, S)

        tgt_mask += self.generate_tgt_mask(tgt.size(1)).to(tgt.device)

        # Adding positional embeddings to src and tgt
        src = (self.src_embeddings(src) * (self.d_model ** 0.5)) + self.positional_embeddings[:src.size(1)]
        tgt = (self.tgt_embeddings(tgt) * (self.d_model ** 0.5)) + self.positional_embeddings[:tgt.size(1)]

        for layer in self.encoder:
            src: torch.Tensor = layer(src, src_mask=src_mask)

        for layer in self.decoder:
            tgt: torch.Tensor = layer(tgt, src, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return tgt