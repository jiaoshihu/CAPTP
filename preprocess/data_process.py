#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2024.02.05
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : main.py

import pickle
import torch
import torch.utils.data as Data


residue2idx = pickle.load(open('./data/AAindex.pkl', 'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_token2index(sequences):
    token2index = residue2idx
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)
    token_index = list()
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_index.append(seq_id)
    return token_index


def pad_sequence(token_list):
    token2index = residue2idx
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i]
        n_pad = 50 - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])
    return data


def construct_dataset(seq_ids):
    if device == "cuda":
        input_ids = torch.cuda.LongTensor(seq_ids)
    else:
        input_ids = torch.LongTensor(seq_ids)

    data_loader = Data.DataLoader(MyDataSet(input_ids),batch_size=128,shuffle=False,drop_last=False)

    return data_loader



class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def load_data(sequence_list):
    token_list = transform_token2index(sequence_list)
    data_test = pad_sequence(token_list)
    data_loader = construct_dataset(data_test)

    return data_loader

