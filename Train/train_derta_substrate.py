import time
import pandas as pd
from Github.Model.DelKcat_substrate import DelKcat_gate
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

# Disable RDKit logs globally to keep training output readable.
RDLogger.DisableLog('rdApp.*')



def getData(data_list, prot_emb, smiles_dict_molt5):
    """
    Convert tabular substrate-pair samples into PyG `Data` objects.

    Each sample corresponds to one protein sequence with two substrates.
    The function attaches:
    - protein features (padded embedding + mask + mean pooled embedding),
    - substrate features for both molecules (MACCS + MolT5 embeddings),
    - regression target `kcat_gt` and metadata fields used in result export.
    """
    output_list = []
    for idx in range(len(data_list)):
        # Unpack one sample: protein index, substrate pair, target value, raw sequence.
        protein_id = data_list[idx][0]
        smiles_1 = data_list[idx][1]
        smiles_2 = data_list[idx][2]
        value = data_list[idx][3]
        seq = data_list[idx][4]

        # Fetch precomputed ProtTrans features for this protein.
        seq_emb, seq_emb_pad, seq_mask = prot_emb[protein_id]
        # Mean-pool token embeddings as a compact sequence-level vector.
        mean_tensor = torch.mean(seq_emb, dim=0, keepdim=True)

        # Build molecular features for substrate 1.
        mol_1 = AllChem.MolFromSmiles(smiles_1)
        fp_1 = MACCSkeys.GenMACCSKeys(mol_1).ToList()
        smiles_emb_molt5_1 = smiles_dict_molt5[smiles_1]
        smiles_emb_molt5_1 = torch.FloatTensor(smiles_emb_molt5_1).unsqueeze(0)

        # Build molecular features for substrate 2.
        mol_2 = AllChem.MolFromSmiles(smiles_2)
        fp_2 = MACCSkeys.GenMACCSKeys(mol_2).ToList()
        smiles_emb_molt5_2 = smiles_dict_molt5[smiles_2]
        smiles_emb_molt5_2 = torch.FloatTensor(smiles_emb_molt5_2).unsqueeze(0)
        # Pack all fields into one graph-like data object for PyG batching.
        output_list.append(Data(smiles_emb_1=torch.FloatTensor(fp_1).unsqueeze(0), seq_emb=seq_emb_pad, mask=seq_mask,
                           kcat_gt=value, smiles_1=smiles_1, seq=seq, mean_seq=mean_tensor,
                           smiles_emb_molt5_1=smiles_emb_molt5_1, smiles_emb_molt5_2=smiles_emb_molt5_2,
                           smiles_emb_2=torch.FloatTensor(fp_2).unsqueeze(0), smiles_2=smiles_2))

    return output_list


def build_sequence_index_map(fasta_file):
    """Create a sequence-string -> index lookup from a FASTA file."""
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(df, fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta',
                 distance=8):
    """
    Build model-ready sample tuples from dataframe rows.

    Rows whose sequence cannot be found in FASTA are skipped and counted.
    """
    data_list = []
    miss_indx = 0
    sequence_map = build_sequence_index_map(fasta_file)
    for index, row in df.iterrows():
        # Map sequence text to ProtTrans feature index.
        protein_id = sequence_map.get(row['Sequence'], -1)
        if protein_id == -1:
            miss_indx +=1
            continue
        # row['kcat_ratio'] = math.log2(row['kcat_ratio'])
        # Keep field order consistent with `getData` expectations.
        data_list.append([protein_id, normalize_smiles(row['Smiles_1']), normalize_smiles(row['Smiles_2']), row['log2_delta_kcat'], row['Sequence']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    return data_list


def kineticsTrainer(arg):
    """
    End-to-end training loop for substrate-pair delta-kcat prediction.

    Pipeline:
    1) Load and filter train/test CSV data.
    2) Convert rows to feature-rich PyG samples.
    3) Train with MSE and evaluate per epoch using R2 + Pearson correlation.
    4) Save best-by-R2 checkpoint and test predictions.
    """
    Epoch = arg.epoch
    batch_size = arg.batch_size
    # Use requested CUDA device if available; otherwise run on CPU.
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{arg.device}')
    else:
        device = torch.device('cpu')
    print(f"used device {device}")

    # Initialize model with the project's original hyperparameter signature.
    model = DelKcat_gate(512, 0, device)
    """Train setting."""
    # Resolve input paths for dataset and sequence index mapping.
    fasta_path = arg.embedding_path + '/sequences_prottrans.fasta'
    train_df = pd.read_csv(arg.train_path)
    test_df = pd.read_csv(arg.test_path)
    test_path = arg.test_path
    # Exclude rows with zero activity values before training/evaluation.
    train_df = train_df[(train_df['Value_1'] != 0) & (train_df['Value_2'] != 0)]
    test_df = test_df[(test_df['Value_1'] != 0) & (test_df['Value_2'] != 0)]
    train_list = read_dataset(train_df, fasta_file=fasta_path, distance=5)
    test_list = read_dataset(test_df, fasta_file=fasta_path, distance=5)
    start_time = time.time()

    # Load precomputed protein and molecule embedding dictionaries.
    prottrans_dict = torch.load(arg.embedding_path + '/prottrans_feature_dict.pt')
    with open(arg.embedding_path + '/SMILES/SMILES_embedding_molT5.pkl', 'rb') as file:
        smiles_dict_molt5 = pickle.load(file)
    print('Parameter loading time:', time.time() - start_time)

    # Build dataset objects and corresponding dataloaders.
    data_graph = os.path.join('../')
    train_set_list = getData(train_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    test_set_list = getData(test_list, prot_emb=prottrans_dict, smiles_dict_molt5=smiles_dict_molt5)
    train_set = MyDataset(data_dir=data_graph, data_df=train_df, graph_type='ConBAP', dis_threshold=5, create=False,
                             dlkcat_feat=train_set_list)
    test_set = MyDataset(data_dir=data_graph, data_df=test_df, graph_type='ConBAP', dis_threshold=5,
                            create=False, dlkcat_feat=test_set_list)

    train_loader = MyDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_kcat = MyDataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


    model = model.to(device)
    # Optimizer, scheduler, and loss remain identical to original behavior.
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-6)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    Loss = nn.MSELoss(reduction='mean')
    # Track recent checkpoint files for simple retention cleanup.
    saved_files = deque(maxlen=2)
    # Best validation/test R2 observed so far.
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
        # Epoch-level training metric.
        train_r2 = r2_score(testY, testPredict)
        print('Training kcat r2: ', train_r2)

        # --------------------
        # Evaluation phase
        # --------------------
        seq_list, smiles_list_1, smiles_list_2, testY, testPredict = [], [], [], [], []
        model.eval()
        for data in tqdm(test_loader_kcat, desc=f"{epoch}"):
            data['dlkcat_features'] = data['dlkcat_features'].to(device)
            testY.extend(data['dlkcat_features'].kcat_gt.cpu().tolist())
            predict = model(data)
            testPredict.extend(predict.cpu().tolist())
            seq_list.extend(data['dlkcat_features'].seq)
            smiles_list_1.extend(data['dlkcat_features'].smiles_1)
            smiles_list_2.extend(data['dlkcat_features'].smiles_2)
        test_r2 = r2_score(testY, testPredict)
        test_pcc = pearsonr(testY, testPredict)
        print('Test kcat r2: ', test_r2)
        print('Test kcat pcc: ', test_pcc)

        # Save outputs only when the current epoch improves best R2.
        if test_r2 >= kcat_start:
            kcat_start = test_r2
            df_predict_test = pd.read_csv(test_path)

            # Write predictions back by exact key: (Sequence, Smiles_1, Smiles_2).
            for i in range(len(smiles_list_1)):
                df_predict_test.loc[(df_predict_test['Sequence'] == seq_list[i]) &
                                    (df_predict_test['Smiles_1'] == smiles_list_1[i]) &
                                    (df_predict_test['Smiles_2'] == smiles_list_2[i]), 'predict kcat'] = testPredict[i]
            df_predict_test.to_csv(save_path + 'Test_sub_gate.csv',
                                   index=False)

            # Save model state with R2 embedded in filename.
            torch.save(model.state_dict(), save_path + 'Deltakcat_sub_' + str(test_r2) + '.pth')
            model_file = save_path + 'Deltakcat_sub_' + str(test_r2) + '.pth'
            saved_files.append(model_file)
            # Delete the oldest tracked checkpoint to limit disk usage.
            if len(saved_files) == 2:
                oldest_file = saved_files.popleft()
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
        # Update LR scheduler once per epoch.
        scheduler.step()
        print('loss', loss_train/len(train_loader))


def parse_args():
    """Parse command-line arguments for training runtime and file paths."""
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, required=False, default=100)
    parser.add_argument('-d', '--device', type=int, required=False, default=5)
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
    # Enable TF32 acceleration on supported NVIDIA GPUs.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    kineticsTrainer(args)
