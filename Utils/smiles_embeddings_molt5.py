import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from os.path import join
import pickle
from tqdm import tqdm
from rdkit.Chem import MACCSkeys


def normalize_smiles(smiles):
    """
    规范化 SMILES 字符串并添加氢原子.

    :param smiles_list: SMILES 字符串的列表
    :return: 规范化后的 SMILES 字符串列表
    """
    normalized = ''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # mol = Chem.AddHs(mol)
            normalized = Chem.MolToSmiles(mol, canonical=True)
        else:
            print(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")

    return normalized


def get_molT5_embed(smiles_list, outpath):
    Molt5_model = "/home/xiongwen/software/molt5-base-smiles2caption"
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
    model = T5EncoderModel.from_pretrained(Molt5_model)
    N_smiles = len(smiles_list)
    smiles_reprs = {}
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in tqdm(smiles_list):
            input_ids = tokenizer(smile, return_tensors="pt").input_ids
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            # seq_emb_pad, seq_mask = pad_and_mask(seq_emb, 1024)
            smiles_reprs[smile] = embed
            final_values.append(embed.reshape(1, -1))
    with open(join(outpath, "SMILES", "SMILES_embedding_molT5"+".pkl"), 'wb') as handle:
        pickle.dump(smiles_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    df = pd.read_csv('/home/xiongwen/DeltaKcat/Data/kkm/kkm_diff_all.csv')
    smiles_list = df['Smiles'].tolist()
    smiles_list = [normalize_smiles(x) for x in smiles_list if isinstance(x, str)]
    smiles_list = sorted(list(set(smiles_list)))
    get_molT5_embed(smiles_list, '/home/xiongwen/DeltaKcat/Embedding/kkm_embedding_all/')