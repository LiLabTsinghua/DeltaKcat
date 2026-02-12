import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from Bio import SeqIO
import numpy as np
import math
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
import random


RDLogger.DisableLog('rdApp.*')


def normalize_smiles(smiles):
    """
    :param smiles_list: SMILES string list
    :return: Normalized SMILES string list
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


def getData(data_list, prot_emb, smiles_dict_molt5):
    output_list = []
    for idx in range(len(data_list)):
        protein_id_1 = data_list[idx][0]
        smiles_1 = data_list[idx][1]
        value = data_list[idx][2]
        seq_1 = data_list[idx][3]
        seq_emb_1, seq_emb_pad_1, seq_mask_1 = prot_emb[protein_id_1]
        mean_tensor_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)
        mol_1 = AllChem.MolFromSmiles(smiles_1)
        fp_1 = MACCSkeys.GenMACCSKeys(mol_1).ToList()
        smiles_emb_molt5_1 = smiles_dict_molt5[smiles_1]
        smiles_emb_molt5_1 = torch.FloatTensor(smiles_emb_molt5_1).unsqueeze(0)

        protein_id_2 = data_list[idx][4]
        smiles_2 = data_list[idx][5]
        seq_2 = data_list[idx][6]
        seq_emb_2, seq_emb_pad_2, seq_mask_2 = prot_emb[protein_id_2]
        mean_tensor_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)
        mol_2 = AllChem.MolFromSmiles(smiles_2)
        fp_2 = MACCSkeys.GenMACCSKeys(mol_2).ToList()
        smiles_emb_molt5_2 = smiles_dict_molt5[smiles_2]
        smiles_emb_molt5_2 = torch.FloatTensor(smiles_emb_molt5_2).unsqueeze(0)
        output_list.append({'data1':Data(smiles_emb=torch.FloatTensor(fp_1).unsqueeze(0), seq_emb=seq_emb_pad_1, mask=seq_mask_1,
                                kcat_gt=value, smiles=smiles_1, seq=seq_1, mean_seq=mean_tensor_1,
                                smiles_emb_molt5=smiles_emb_molt5_1),
                            'data2':Data(smiles_emb=torch.FloatTensor(fp_2).unsqueeze(0), seq_emb=seq_emb_pad_2, mask=seq_mask_2,
                                kcat_gt=value, smiles=smiles_2, seq=seq_2, mean_seq=mean_tensor_2,
                                smiles_emb_molt5=smiles_emb_molt5_2)})
    return output_list


def build_sequence_index_map(fasta_file):
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(path='../Code/my_data/eitlem_kcat_km.csv', fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta'):
    df = pd.read_csv(path)
    data_list = []
    miss_indx = 0
    sequence_map = build_sequence_index_map(fasta_file)
    for index, row in df.iterrows():
        protein_id_1 = sequence_map.get(row['Sequence_1'], -1)
        if protein_id_1 == -1:
            miss_indx +=1
            continue

        protein_id_2 = sequence_map.get(row['Sequence_2'], -1)
        if protein_id_2 == -1:
            miss_indx += 1
            continue
        # row['Value'] = math.log2(row['Value'])
        data_list.append([protein_id_1, normalize_smiles(row['Smiles']), row['log10_Value'], row['Sequence_1'],
                          protein_id_2, normalize_smiles(row['Smiles']), row['Sequence_2']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    print('Number of missed sequences:', miss_indx)
    return data_list
