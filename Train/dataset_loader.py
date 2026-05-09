import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from Bio import SeqIO
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
from torch_geometric.data.batch import Batch
import warnings
from threading import Lock


# Reserved lock for potential multi-threaded shared-resource protection.
my_lock = Lock()
# Silence RDKit logs to reduce noisy console output during training.
RDLogger.DisableLog('rdApp.*')
# Print full NumPy arrays for easier debugging.
np.set_printoptions(threshold=np.inf)
# Suppress warnings to keep logs concise.
warnings.filterwarnings('ignore')


# %%
class MyDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        # Reuse the dataset-level collate function for custom batch assembly.
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#


class MyDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_df, num_process=8, create=False, dlkcat_feat=None):
        # Store dataset configuration and references.
        self.data_df = data_df
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()
        self.dlkcat_feat = dlkcat_feat

    def _pre_process(self):
        # Preprocess hook (currently keeps only local references).
        data_df = self.data_df


    def __getitem__(self, idx):
        # Return the same feature for input/target pairing.
        seq_feature = self.dlkcat_feat[idx]
        return seq_feature, seq_feature

    def collate_fn(self, data_list):
        # Pack graph samples into one PyG batch for model input.
        batch_dlkcat_feat = Batch.from_data_list([data[0] for data in data_list])
        batch = {'dlkcat_features': batch_dlkcat_feat}
        return batch

    def __len__(self):
        # Dataset length follows the dataframe row count.
        return len(self.data_df)


def normalize_smiles(smiles):
    """
    :param smiles_list: SMILES string list
    :return: Normalized SMILES string list
    """
    # Return an empty string when parsing fails.
    normalized = ''
    try:
        # Convert to canonical SMILES for normalized molecule representation.
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
    # Build paired PyG Data objects from raw sample tuples.
    output_list = []
    for idx in range(len(data_list)):
        # First item in the pair: protein index, SMILES, label, and sequence.
        protein_id_1 = data_list[idx][0]
        smiles_1 = data_list[idx][1]
        value = data_list[idx][2]
        seq_1 = data_list[idx][3]
        # Protein features: token embeddings, padded embeddings, and mask.
        seq_emb_1, seq_emb_pad_1, seq_mask_1 = prot_emb[protein_id_1]
        # Mean-pool token embeddings to a sequence-level vector.
        mean_tensor_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)
        # Generate MACCS fingerprint from the molecule.
        mol_1 = AllChem.MolFromSmiles(smiles_1)
        fp_1 = MACCSkeys.GenMACCSKeys(mol_1).ToList()
        # Load MolT5 molecular embedding.
        smiles_emb_molt5_1 = smiles_dict_molt5[smiles_1]
        smiles_emb_molt5_1 = torch.FloatTensor(smiles_emb_molt5_1).unsqueeze(0)

        # Second item in the pair.
        protein_id_2 = data_list[idx][4]
        smiles_2 = data_list[idx][5]
        seq_2 = data_list[idx][6]
        seq_emb_2, seq_emb_pad_2, seq_mask_2 = prot_emb[protein_id_2]
        mean_tensor_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)
        mol_2 = AllChem.MolFromSmiles(smiles_2)
        fp_2 = MACCSkeys.GenMACCSKeys(mol_2).ToList()
        smiles_emb_molt5_2 = smiles_dict_molt5[smiles_2]
        smiles_emb_molt5_2 = torch.FloatTensor(smiles_emb_molt5_2).unsqueeze(0)
        # data1/data2 share the same regression target (value).
        output_list.append({'data1':Data(smiles_emb=torch.FloatTensor(fp_1).unsqueeze(0), seq_emb=seq_emb_pad_1, mask=seq_mask_1,
                                kcat_gt=value, smiles=smiles_1, seq=seq_1, mean_seq=mean_tensor_1,
                                smiles_emb_molt5=smiles_emb_molt5_1),
                            'data2':Data(smiles_emb=torch.FloatTensor(fp_2).unsqueeze(0), seq_emb=seq_emb_pad_2, mask=seq_mask_2,
                                kcat_gt=value, smiles=smiles_2, seq=seq_2, mean_seq=mean_tensor_2,
                                smiles_emb_molt5=smiles_emb_molt5_2)})
    return output_list


def build_sequence_index_map(fasta_file):
    # Map each FASTA sequence string to a continuous index.
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def read_dataset(path='../Code/my_data/eitlem_kcat_km.csv', fasta_file='../Data/kcat_prottrans/sequences_prottrans.fasta'):
    # Read CSV data and map sequence text to FASTA indices.
    df = pd.read_csv(path)
    data_list = []
    miss_indx = 0
    sequence_map = build_sequence_index_map(fasta_file)
    for index, row in df.iterrows():
        # Skip and count sequences not found in FASTA map.
        protein_id_1 = sequence_map.get(row['Sequence_1'], -1)
        if protein_id_1 == -1:
            miss_indx +=1
            continue

        protein_id_2 = sequence_map.get(row['Sequence_2'], -1)
        if protein_id_2 == -1:
            miss_indx += 1
            continue
        # Sample layout: protein1 + molecule + label + seq1 + protein2 + molecule + seq2.
        data_list.append([protein_id_1, normalize_smiles(row['Smiles']), row['log10_Value'], row['Sequence_1'],
                          protein_id_2, normalize_smiles(row['Smiles']), row['Sequence_2']])
    random.seed(40)  # Set the random seed to ensure reproducible results.
    random.shuffle(data_list)  # Shuffle the data list.
    print('Number of missed sequences:', miss_indx)
    return data_list
