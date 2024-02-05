#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2024.02.05
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : main.py



import torch
import torch.nn as nn
import numpy as np
import math




max_len =51
n_layers = 1
n_head = 8
d_model = 256
d_ff = 1024
d_k = 32
d_v = 32
vocab_size = 24




class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)

    def forward(self, input_ids):
        x = self.src_emb(input_ids)
        embeddings = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        return embeddings


class ConvMod(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.a = nn.Sequential(
                nn.Conv1d(d_model, d_model, 1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        )

        self.v = nn.Conv1d(d_model, d_model, 1)
        self.proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x.transpose(1, 2))
        x = a * self.v(x.transpose(1, 2))
        x = self.proj(x)

        return x


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Conv1d(d_model, d_ff, 1)
        self.pos = nn.Conv1d(d_ff, d_ff, 3, padding=1, groups=d_ff)
        self.fc2 = nn.Conv1d(d_ff, d_model, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x.transpose(1, 2))
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs, attention_map


class CAPTP(nn.Module):
    def __init__(self):
        super(CAPTP, self).__init__()

        self.emb = EmbeddingLayer(vocab_size, d_model)
        self.cnnatt = ConvMod(d_model)
        self.cnnff = MLP(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.5)

        self.pool = MaxPooling()


        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.block1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), )

        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2))


    def forward(self, input_ids):
        attention_mask = (input_ids != 0).float()
        emb_out = self.emb(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)

        cnnatt_output = self.cnnatt(emb_out).transpose(1, 2)
        cnnatt_output = self.dropout(cnnatt_output)
        cnnatt_output = self.norm(emb_out + cnnatt_output)

        cnnff_output = self.cnnff(cnnatt_output).transpose(1, 2)
        cnnff_output = self.dropout(cnnff_output)
        cnn_output = self.norm(cnnatt_output + cnnff_output)

        for layer in self.layers:
            enc_output, attention_values = layer(cnn_output, enc_self_attn_mask)

        enc_seq = enc_output[:,1:,:]
        pooled_output = self.pool(enc_seq, attention_mask[:,1:])
        representations = self.block1(pooled_output)

        return representations

    
    def get_logits(self, input_ids):
        with torch.no_grad():
            output = self.forward(input_ids)
        logits = self.block2(output)

        return logits
