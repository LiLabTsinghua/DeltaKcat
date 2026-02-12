# %%
import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, is_aa
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data.batch import Batch
import warnings
from threading import Lock
my_lock = Lock()
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')


# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=8, graph_type='ConBAP', num_process=8, create=False, dlkcat_feat=None):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()
        self.dlkcat_feat = dlkcat_feat

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df


    def __getitem__(self, idx):
        seq_feature = self.dlkcat_feat[idx]
        return seq_feature, seq_feature

    def collate_fn(self, data_list):
        batch_dlkcat_feat = Batch.from_data_list([data[0] for data in data_list])
        batch = {'dlkcat_features': batch_dlkcat_feat}
        return batch

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':
    print('finish!')

