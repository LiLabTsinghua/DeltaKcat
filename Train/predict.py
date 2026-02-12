import time
import pandas as pd
from torch import nn
import torch
from Model.DelKcat_sequence import DelKcat_gate
# from dlkcat2_new import DLKcat2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from dataset_loader import read_dataset, getData, normalize_smiles
from dataset_GVP import GraphDataset, PLIDataLoader
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim.lr_scheduler as lr_scheduler
import pickle
from collections import deque
import argparse
from rdkit import Chem
import os
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
from Bio import SeqIO
import math
import random
import torch.distributed as dist

RDLogger.DisableLog('rdApp.*')

class CombinedLoss(nn.Module):
    """
    total = alpha * MSE(y1_pred, y1_true) + beta * MSE(y2_pred, y2_true)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, reductions: str = "mean"):
        super().__init__()
        self.mse1 = nn.MSELoss(reduction=reductions)
        self.mse2 = nn.MSELoss(reduction=reductions)
        self.alpha = alpha
        self.beta = beta

    def forward(self,y1_pred: torch.Tensor, y1_true: torch.Tensor, y2_pred: torch.Tensor, y2_true: torch.Tensor):
        loss1 = self.mse1(y1_pred, y1_true)
        loss2 = self.mse2(y2_pred, y2_true)
        total = self.alpha * loss1 + self.beta * loss2
        return total


class ListMLELoss(nn.Module):
    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 按真实值排序
        _, indices = torch.sort(y_true, descending=True)
        y_pred_sorted = y_pred[indices]
        # ListMLE公式
        max_pred = torch.max(y_pred_sorted)
        y_pred_sorted = y_pred_sorted - max_pred  # 防止溢出
        cumsums = torch.logcumsumexp(y_pred_sorted.flip(dims=[0]), dim=0).flip(dims=[0])
        loss = torch.sum(cumsums - y_pred_sorted)
        return loss


def getData(data_list, prot_emb, smiles_dict_molt5):
    output_list = []
    for idx in range(len(data_list)):
        protein_id_1 = data_list[idx][0]
        protein_id_2 = data_list[idx][1]
        smiles= data_list[idx][2]
        value = data_list[idx][3]
        seq_1 = data_list[idx][4]
        seq_2 = data_list[idx][5]
        seq_emb_1, seq_emb_pad_1, seq_mask_1 = prot_emb[protein_id_1]
        mean_tensor_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)

        seq_emb_2, seq_emb_pad_2, seq_mask_2 = prot_emb[protein_id_2]
        mean_tensor_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)

        mol = AllChem.MolFromSmiles(smiles)
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
        smiles_emb_molt5 = smiles_dict_molt5[smiles]
        smiles_emb_molt5 = torch.FloatTensor(smiles_emb_molt5).unsqueeze(0)
        output_list.append(
            Data(smiles_emb=torch.FloatTensor(fp).unsqueeze(0), seq_emb_1=seq_emb_pad_1, mask_1=seq_mask_1,
                 kcat_gt=value, smiles=smiles, seq_1=seq_1, mean_seq_1=mean_tensor_1,
                 smiles_emb_molt5=smiles_emb_molt5, seq_2=seq_2, mean_seq_2=mean_tensor_2,
                 seq_emb_2=seq_emb_pad_2, mask_2=seq_mask_2))

    return output_list


def build_sequence_index_map(fasta_file):
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(df, fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta',
                 distance=8):

    # df = df[df['ECNumber'].str.startswith('1')]
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
        # row['kcat_ratio'] = math.log2(row['kcat_ratio'])
        data_list.append([protein_id_1, protein_id_2, normalize_smiles(row['Smiles']), row['log2_delta_kcat'], row['Sequence_1'], row['Sequence_2']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    print('Number of missed sequences:', miss_indx)
    return data_list


def kineticsTrainer(arg):
    Epoch = arg.epoch
    batch_size = arg.batch_size
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{arg.device}')
    else:
        device = torch.device('cpu')
    print(f"used device {device}")

    model = DelKcat_gate(167, 512, 1280, 1, 0, 10, device)
    """Train setting."""

    fasta_path = '../Embedding/pair_seq_kcat/sequences_prottrans.fasta'
    test_df = pd.read_csv('../DelKcat/Data/pair_seq_kcat2.csv')
    test_path = '../Data/pair_seq_kcat2.csv'

    test_df = test_df[(test_df['Value_1'] != 0) & (test_df['Value_2'] != 0)]
    test_list = read_dataset(test_df, fasta_file=fasta_path, distance=5)



    start_time = time.time()

    prottrans_dict = torch.load('../Embedding/embedding_pair_seq_kcat/prottrans_feature_dict.pt')
    with open('../Embedding/embedding_pair_seq_kcat/SMILES/SMILES_embedding_molT5.pkl', 'rb') as file:
        smiles_dict_molt5 = pickle.load(file)
    print('Parameter loading time:', time.time() - start_time)

    data_graph = os.path.join('../')
    test_set_list = getData(test_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    test_set = GraphDataset(data_dir=data_graph, data_df=test_df, graph_type='ConBAP', dis_threshold=5,
                            create=False, dlkcat_feat=test_set_list)

    test_loader_kcat = PLIDataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


    model = model.to(device)
    # model.load_state_dict(torch.load('/home/xiongwen/DelKcat/Result/kcat/Delta_kcat_seq_gate_random_0.6393588535279087.pth'))
    model.load_state_dict(
        torch.load('../Result/kkm/Delta_kkm_seq_gate_random_0.5585063310728693.pth'))

    seq_list_1, seq_list_2, smiles_list, testY, testPredict = [], [], [], [], []
    model.eval()
    for data in tqdm(test_loader_kcat):
        data['dlkcat_features'] = data['dlkcat_features'].to(device)
        testY.extend(data['dlkcat_features'].kcat_gt.cpu().tolist())
        predict = model(data)
        testPredict.extend(predict.cpu().tolist())
        seq_list_1.extend(data['dlkcat_features'].seq_1)
        seq_list_2.extend(data['dlkcat_features'].seq_2)
        smiles_list.extend(data['dlkcat_features'].smiles)
    test_r2 = r2_score(testY, testPredict)
    test_pcc = pearsonr(testY, testPredict)
    test_scc = spearmanr(testY, testPredict)
    print('Test kcat pcc: ', test_pcc)
    print('Test kcat scc: ', test_scc)

    df_predict_test = pd.read_csv(test_path)

    for i in range(len(smiles_list)):
        df_predict_test.loc[(df_predict_test['Sequence_1'] == seq_list_1[i]) &
                            (df_predict_test['Sequence_2'] == seq_list_2[i]), 'predict kcat'] = testPredict[i]

    df_predict_test.to_csv('../Data/delta_kkm_seq_gate.csv',
                           index=False)
    print(df_predict_test['predict kcat'])



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, required=False, default=100)
    parser.add_argument('-d', '--device', type=int, required=False, default=5)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64)
    parser.add_argument('-l', '--lr', type=int, required=False, default=5e-5)
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    kineticsTrainer(args)