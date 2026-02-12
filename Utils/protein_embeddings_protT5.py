import os
from os.path import join
from Bio import SeqIO
import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re

def create_file(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        all_files = os.listdir(path)
        for file in all_files:
            os.remove(join(path, file))


def create_fasta_file(sequences, filename):
    ofile = open(filename, "w")
    for k, seq in enumerate(sequences):
        ofile.write(">" + str(k) + "\n" + seq + "\n")
    ofile.close()


def build_sequence_index_map(fasta_file):
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def protein_embedding_prottrans(sequences, output_path='/home/xiongwen/new_dlkcat/Data/kcat_prottrans/', device='cuda:4'):
    fasta_file = join(output_path, "sequences_prottrans.fasta")
    protein_file = join(output_path, "Protein")
    create_fasta_file(sequences, fasta_file)
    sequence_map = build_sequence_index_map(fasta_file)
    try:
        create_file(protein_file)
    except IsADirectoryError:
        print('File already exists')
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('/home/xiongwen/software/prot_t5_xl_uniref50', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("/home/xiongwen/software/prot_t5_xl_uniref50")
    model.eval()
    if torch.cuda.is_available():
        model = model.to(device)
        print("Transferred model to GPU")
    label_list = [str(sequence_map.get(x, -1)) for x in sequences]
    output_dir = join(output_path, "Protein", "temp")
    create_file(output_dir)
    # seq_list = [x for x in seq_list if len(x) < 1024]

    for i in range(len(sequences)):
        if len(sequences[i]) > 1024:
            sequences[i] = sequences[i][:512] + sequences[i][-512:]
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]


    for i in tqdm(range(len(sequences))):
        print('For sequence ', str(i + 1))
        output_file = join(output_dir, label_list[i] + ".pt")
        sequences_Example_i = sequences[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        torch.save(embedding, output_file)


def build_sequence_index_map(fasta_file):
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def save_prottrans_features(seq_list, filepath):
    # seq_list = list(set(df['Sequence'].tolist()))
    seq_list = [x for x in seq_list if isinstance(x, str)]
    seq_path = os.path.join(filepath + f'/Protein/temp/', '{}.pt')
    sequence_map = build_sequence_index_map(filepath + 'sequences_prottrans.fasta')
    esm_dict = {}
    for seq in tqdm(seq_list):
        pro_id = sequence_map.get(seq, -1)
        seq_emb = torch.load(seq_path.format(pro_id), weights_only=False)
        seq_emb = torch.tensor(seq_emb).squeeze(0)
        seq_emb = seq_emb[:seq_emb.shape[0] - 1, :]
        seq_emb_pad, seq_mask = pad_and_mask(seq_emb, 1024)
        esm_dict[pro_id] = [seq_emb, seq_emb_pad, seq_mask]
    torch.save(esm_dict, filepath + 'prottrans_feature_dict.pt')


def pad_and_mask(emb, max_len):
    if len(emb) < max_len:
        padded_emb = np.pad(emb, ((0, max_len - len(emb)), (0, 0)), mode='constant', constant_values=0)
        mask = np.concatenate([np.ones(len(emb)), np.zeros(max_len - len(emb))])
    else:
        padded_emb = emb[:max_len]
        mask = np.ones(max_len)
    return torch.FloatTensor(padded_emb), torch.BoolTensor(mask)


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
            mol = Chem.AddHs(mol)
            normalized = Chem.MolToSmiles(mol, canonical=True)
        else:
            print(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
    return normalized


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    df = pd.read_csv('/home/xiongwen/DeltaKcat/Data/kkm/kkm_diff_all.csv')
    out_dir = '/home/xiongwen/DeltaKcat/Embedding/kkm_embedding_all/'
    seq_list = sorted(list(set(df['Sequence'].tolist())))
    seq_list = [x for x in seq_list if isinstance(x, str)]

    # protein_embedding_prottrans(seq_list, out_dir)
    save_prottrans_features(seq_list, out_dir)