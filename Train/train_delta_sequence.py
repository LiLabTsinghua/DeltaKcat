import time
import pandas as pd
from Model.DelKcat_sequence import DelKcat_gate
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from dataset_loader import normalize_smiles, MyDataset, MyDataLoader
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

# Silence RDKit logs globally to avoid cluttering training output.
RDLogger.DisableLog('rdApp.*')


def getData(data_list, prot_emb, smiles_dict_molt5):
    """
    Convert tabular paired-sequence records into PyG `Data` objects.

    Each output sample contains:
    - sequence embeddings/masks for both proteins in a pair,
    - one shared molecular representation (MACCS + MolT5),
    - regression label (`kcat_gt`) and metadata fields used for evaluation export.
    """
    output_list = []
    for idx in range(len(data_list)):
        # Unpack one pair sample: two protein indices, one molecule, one target value.
        protein_id_1 = data_list[idx][0]
        protein_id_2 = data_list[idx][1]
        smiles= data_list[idx][2]
        value = data_list[idx][3]
        seq_1 = data_list[idx][4]
        seq_2 = data_list[idx][5]

        # Retrieve ProtTrans features for sequence 1:
        # token-level embedding, padded embedding, and attention mask.
        seq_emb_1, seq_emb_pad_1, seq_mask_1 = prot_emb[protein_id_1]
        # Mean-pool token embeddings to a single sequence descriptor.
        mean_tensor_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)

        # Retrieve and pool sequence 2 features.
        seq_emb_2, seq_emb_pad_2, seq_mask_2 = prot_emb[protein_id_2]
        mean_tensor_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)

        # Build molecular features from SMILES:
        # MACCS keys are used as a compact handcrafted fingerprint.
        mol = AllChem.MolFromSmiles(smiles)
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
        # MolT5 embedding is loaded from a precomputed dictionary.
        smiles_emb_molt5 = smiles_dict_molt5[smiles]
        smiles_emb_molt5 = torch.FloatTensor(smiles_emb_molt5).unsqueeze(0)

        # Store all paired fields in one graph-style container for batching.
        output_list.append(
            Data(smiles_emb=torch.FloatTensor(fp).unsqueeze(0), seq_emb_1=seq_emb_pad_1, mask_1=seq_mask_1,
                 kcat_gt=value, smiles=smiles, seq_1=seq_1, mean_seq_1=mean_tensor_1,
                 smiles_emb_molt5=smiles_emb_molt5, seq_2=seq_2, mean_seq_2=mean_tensor_2,
                 seq_emb_2=seq_emb_pad_2, mask_2=seq_mask_2))

    return output_list


def build_sequence_index_map(fasta_file):
    """Map each FASTA sequence string to its integer index in parse order."""
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(df, fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta'):
    """
    Build training/evaluation tuples from a dataframe.

    Rows with sequences missing in the FASTA index are skipped.
    """
    data_list = []
    miss_indx = 0
    sequence_map = build_sequence_index_map(fasta_file)
    for index, row in df.iterrows():
        # Resolve Sequence_1 to the embedding index space.
        protein_id_1 = sequence_map.get(row['Sequence_1'], -1)
        if protein_id_1 == -1:
            miss_indx +=1
            continue
        # Resolve Sequence_2 to the embedding index space.
        protein_id_2 = sequence_map.get(row['Sequence_2'], -1)
        if protein_id_2 == -1:
            miss_indx += 1
            continue
        # Keep the sample layout expected by `getData`.
        data_list.append([protein_id_1, protein_id_2, normalize_smiles(row['Smiles']), row['log2_delta_kcat'], row['Sequence_1'], row['Sequence_2']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    return data_list


def Trainer(arg):
    """
    End-to-end training routine for the sequence-gated Delta-kcat model.

    Flow:
    1) Load CSVs and precomputed embeddings.
    2) Convert records into graph-like batched tensors.
    3) Train with MSE loss, evaluate with R2/Pearson each epoch.
    4) Save best-by-R2 checkpoint and corresponding test predictions.
    """
    Epoch = arg.epoch
    batch_size = arg.batch_size
    # Select user-specified CUDA device when available.
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{arg.device}')
    else:
        device = torch.device('cpu')
    print(f"used device {device}")

    # Model dimensions/configuration follow the original project setup.
    model = DelKcat_gate(512,0, device)
    """Train setting."""

    # Input files: paired train/test CSVs and FASTA used for sequence indexing.
    fasta_path = arg.embedding_path + '/sequences_prottrans.fasta'
    train_df = pd.read_csv(arg.train_path)
    test_df = pd.read_csv(arg.test_path)
    test_path = arg.test_path

    # Remove zero-value rows before building pair samples.
    train_df = train_df[(train_df['Value_1'] != 0) & (train_df['Value_2'] != 0)]
    test_df = test_df[(test_df['Value_1'] != 0) & (test_df['Value_2'] != 0)]
    train_list = read_dataset(train_df, fasta_file=fasta_path)
    test_list = read_dataset(test_df, fasta_file=fasta_path)
    start_time = time.time()

    # Load precomputed protein and molecule embeddings.
    prottrans_dict = torch.load(arg.embedding_path + '/prottrans_feature_dict.pt')
    with open(arg.embedding_path + '/SMILES/SMILES_embedding_molT5.pkl', 'rb') as file:
        smiles_dict_molt5 = pickle.load(file)
    print('Parameter loading time:', time.time() - start_time)

    # Wrap parsed samples into dataset/dataloader objects expected by the model code.
    train_set_list = getData(train_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    test_set_list = getData(test_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    train_set = MyDataset(data_df=train_df, create=False,
                             dlkcat_feat=train_set_list)
    test_set = MyDataset(data_df=test_df,
                            create=False, dlkcat_feat=test_set_list)

    train_loader = MyDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_kcat = MyDataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


    model = model.to(device)
    # Optimizer/scheduler/loss follow the original recipe.
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-6)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    Loss = nn.MSELoss(reduction='mean')
    # Keep only latest best checkpoints (deque size 2 before pruning logic).
    saved_files = deque(maxlen=2)
    # Tracks best test R2 so far.
    kcat_start = 0
    save_path = arg.output_path
    for epoch in range(1, Epoch + 1):
        # --------------------
        # Training phase
        # --------------------
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
        # Report epoch-level training R2.
        train_r2 = r2_score(testY, testPredict)
        print('Training kcat r2: ', train_r2)

        # --------------------
        # Evaluation phase
        # --------------------
        seq_list_1, seq_list_2, smiles_list, testY, testPredict = [], [], [], [], []
        model.eval()
        for data in tqdm(test_loader_kcat, desc=f"{epoch}"):
            data['dlkcat_features'] = data['dlkcat_features'].to(device)
            testY.extend(data['dlkcat_features'].kcat_gt.cpu().tolist())
            predict = model(data)
            testPredict.extend(predict.cpu().tolist())
            seq_list_1.extend(data['dlkcat_features'].seq_1)
            seq_list_2.extend(data['dlkcat_features'].seq_2)
            smiles_list.extend(data['dlkcat_features'].smiles)
        test_r2 = r2_score(testY, testPredict)
        test_pcc = pearsonr(testY, testPredict)
        print('Test kcat r2: ', test_r2)
        print('Test kcat pcc: ', test_pcc)

        # Save outputs only when this epoch improves the best test R2.
        if test_r2 >= kcat_start:
            kcat_start = test_r2
            df_predict_test = pd.read_csv(test_path)

            # Write predictions back by exact (seq1, seq2, smiles) key matching.
            for i in range(len(smiles_list)):
                df_predict_test.loc[(df_predict_test['Sequence_1'] == seq_list_1[i]) &
                                    (df_predict_test['Sequence_2'] == seq_list_2[i]) &
                                    (df_predict_test['Smiles'] == smiles_list[i]), 'predict kcat'] = testPredict[i]
            df_predict_test.to_csv(save_path + 'Test_seq_gate.csv',
                                   index=False)

            # Save model checkpoint tagged with current best R2.
            torch.save(model.state_dict(), save_path + '/Deltakcat_seq_' + str(test_r2) + '.pth')
            model_file = save_path + '/Deltakcat_seq_' + str(test_r2) + '.pth'
            saved_files.append(model_file)
            # Keep storage bounded by deleting the oldest tracked checkpoint.
            if len(saved_files) == 2:
                oldest_file = saved_files.popleft()
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
        # Step LR scheduler once per epoch.
        scheduler.step()
        print('loss', loss_train/len(train_loader))


def parse_args():
    """Parse CLI arguments for training and I/O paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, required=False, default=100)
    parser.add_argument('-d', '--device', type=int, required=False, default=0)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64)
    parser.add_argument('-l', '--lr', type=int, required=False, default=5e-5)
    parser.add_argument('-tr', '--train_path', type=str, required=False,
                        default='/path/train_data.csv')
    parser.add_argument('-te', '--test_path', type=str, required=False,
                        default='/path/test_data.csv')
    parser.add_argument('-e', '--embedding_path', type=str, required=False,
                        default='/path/embedding_path/')
    parser.add_argument('-o', '--output_path', type=str, required=False,
                        default='/path/output_path/')
    return parser.parse_args()


if __name__ == '__main__':
    # Enable TF32 math on supported NVIDIA GPUs for faster matrix operations.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    Trainer(args)
