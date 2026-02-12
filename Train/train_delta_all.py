import time
import pandas as pd
from Model.DelKcat_all import DelKcat_gate
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from dataset_loader import read_dataset, getData, normalize_smiles
from dataset_GVP import GraphDataset, PLIDataLoader
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim.lr_scheduler as lr_scheduler
import pickle
from collections import deque
import argparse
import os
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
from Bio import SeqIO
import random

RDLogger.DisableLog('rdApp.*')


def getData(data_list, prot_emb, smiles_dict_molt5):
    output_list = []
    for idx in range(len(data_list)):
        protein_id_1 = data_list[idx][0]
        protein_id_2 = data_list[idx][1]
        smiles_1= data_list[idx][2]
        smiles_2 = data_list[idx][3]
        value = data_list[idx][4]
        seq_1 = data_list[idx][5]
        seq_2 = data_list[idx][6]
        seq_emb_1, seq_emb_pad_1, seq_mask_1 = prot_emb[protein_id_1]
        mean_tensor_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)
        seq_emb_2, seq_emb_pad_2, seq_mask_2 = prot_emb[protein_id_2]
        mean_tensor_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)
        mol_1 = AllChem.MolFromSmiles(smiles_1)
        fp_1 = MACCSkeys.GenMACCSKeys(mol_1).ToList()
        smiles_emb_molt5_1 = smiles_dict_molt5[smiles_1]
        smiles_emb_molt5_1 = torch.FloatTensor(smiles_emb_molt5_1).unsqueeze(0)
        mol_2 = AllChem.MolFromSmiles(smiles_2)
        fp_2 = MACCSkeys.GenMACCSKeys(mol_2).ToList()
        smiles_emb_molt5_2 = smiles_dict_molt5[smiles_2]
        smiles_emb_molt5_2 = torch.FloatTensor(smiles_emb_molt5_2).unsqueeze(0)
        output_list.append(
            Data(smiles_emb_1=torch.FloatTensor(fp_1).unsqueeze(0),
                 smiles_emb_2=torch.FloatTensor(fp_2).unsqueeze(0), seq_emb_1=seq_emb_pad_1, mask_1=seq_mask_1,
                 kcat_gt=value, smiles_1=smiles_1, smiles_2=smiles_2, seq_1=seq_1, mean_seq_1=mean_tensor_1,
                 smiles_emb_molt5_1=smiles_emb_molt5_1, smiles_emb_molt5_2=smiles_emb_molt5_2,
                 seq_2=seq_2, mean_seq_2=mean_tensor_2,
                 seq_emb_2=seq_emb_pad_2, mask_2=seq_mask_2))

    return output_list


def build_sequence_index_map(fasta_file):
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(df, fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta'):
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
        data_list.append([protein_id_1, protein_id_2, normalize_smiles(row['Smiles_1']),
                          normalize_smiles(row['Smiles_2']), row['log2_delta_kcat'], row['Sequence_1'], row['Sequence_2']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    print('Number of missed sequences:', miss_indx)
    return data_list


def Trainer(arg):
    Epoch = arg.epoch
    batch_size = arg.batch_size
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{arg.device}')
    else:
        device = torch.device('cpu')
    print(f"used device {device}")

    model = DelKcat_gate(512, 0, device)
    """Train setting."""

    fasta_path = '../Embedding/kkm_embedding_all/sequences_prottrans.fasta'
    train_df = pd.read_csv('../Data/kkm_pair_all/Train_pair_all_kkm_random.csv')
    test_df = pd.read_csv('../Data/kkm_pair_all/Test_pair_all_kkm_random.csv')

    train_df = train_df[(train_df['Value_1'] != 0) & (train_df['Value_2'] != 0)]
    test_df = test_df[(test_df['Value_1'] != 0) & (test_df['Value_2'] != 0)]
    print(len(train_df), len(test_df))
    train_list = read_dataset(train_df, fasta_file=fasta_path)
    test_list = read_dataset(test_df, fasta_file=fasta_path)

    start_time = time.time()
    prottrans_dict = torch.load('../Embedding/kkm_embedding_all/prottrans_feature_dict.pt')
    with open('../Embedding/kkm_embedding_all/SMILES/SMILES_embedding_molT5.pkl', 'rb') as file:
        smiles_dict_molt5 = pickle.load(file)

    print('Parameter loading time:', time.time() - start_time)
    data_graph = os.path.join('../')
    train_set_list = getData(train_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    test_set_list = getData(test_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    train_set = GraphDataset(data_dir=data_graph, data_df=train_df, graph_type='ConBAP', dis_threshold=5, create=False,
                             dlkcat_feat=train_set_list)
    test_set = GraphDataset(data_dir=data_graph, data_df=test_df, graph_type='ConBAP', dis_threshold=5, create=False,
                            dlkcat_feat=test_set_list)

    train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_kcat = PLIDataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-6)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    Loss = nn.MSELoss(reduction='mean')
    saved_files = deque(maxlen=2)
    kcat_start = 0
    save_path = '../Result/kkm/'
    for epoch in range(1, Epoch + 1):
        model.train()
        loss_train = 0
        num_batches = 0
        testY, testPredict = [], []
        for data in tqdm(train_loader, desc=f"epoch {epoch} train:", leave=False):
            optimizer.zero_grad()
            data['dlkcat_features'] = data['dlkcat_features'].to(device)

            testY.extend(data['dlkcat_features'].kcat_gt.cpu().tolist())

            predict = model(data)
            testPredict.extend(predict.cpu().tolist())
            loss = Loss(predict, data['dlkcat_features'].kcat_gt)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            num_batches += 1
        train_r2 = r2_score(testY, testPredict)
        print('Training kcat r2: ', train_r2)

        seq_list_1, seq_list_2, smiles_list_1, smiles_list_2, testY, testPredict = [], [], [], [], [], []
        model.eval()
        for data in tqdm(test_loader_kcat, desc=f"{epoch}"):
            data['dlkcat_features'] = data['dlkcat_features'].to(device)
            testY.extend(data['dlkcat_features'].kcat_gt.cpu().tolist())
            predict = model(data)
            testPredict.extend(predict.cpu().tolist())
            seq_list_1.extend(data['dlkcat_features'].seq_1)
            seq_list_2.extend(data['dlkcat_features'].seq_2)
            smiles_list_1.extend(data['dlkcat_features'].smiles_1)
            smiles_list_2.extend(data['dlkcat_features'].smiles_2)
        test_r2 = r2_score(testY, testPredict)
        test_pcc = pearsonr(testY, testPredict)
        print('Test kcat r2: ', test_r2)
        print('Test kcat pcc: ', test_pcc)

        if test_r2 >= kcat_start:
            kcat_start = test_r2
            torch.save(model.state_dict(), save_path + 'Delta_kkm_all_gate_random_' + str(test_r2) + '.pth')
            model_file = save_path + 'Delta_kkm_all_gate_random_' + str(test_r2) + '.pth'
            saved_files.append(model_file)
            if len(saved_files) == 2:
                oldest_file = saved_files.popleft()
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
        scheduler.step()
        print('loss', loss_train/len(train_loader))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, required=False, default=100)
    parser.add_argument('-d', '--device', type=int, required=False, default=4)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64)
    parser.add_argument('-l', '--lr', type=int, required=False, default=5e-5)
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    Trainer(args)