"""Generate MolT5 embeddings for SMILES and save them to a pickle file.

Typical flow:
1) Read SMILES strings from a dataset.
2) Canonicalize each SMILES with RDKit.
3) Encode with MolT5 and mean-pool token embeddings.
4) Save mapping {smiles: embedding_vector}.
"""

import argparse
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


# Canonicalize a raw SMILES string so downstream embedding lookup is consistent.
def normalize_smiles(smiles):
    """
    Normalize the SMILES string and add hydrogen atoms.
        :param smiles_list: List of SMILES strings
        :return: List of normalized SMILES strings
    """
    # Returns empty string if parsing fails; caller may filter it later.
    normalized = ''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            normalized = Chem.MolToSmiles(mol, canonical=True)
        else:
            print(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")

    return normalized


def get_molT5_embed(smiles_list, output_dir, model_dir):
    """Encode SMILES with MolT5 and store pooled embeddings in a pickle file.

    Args:
        smiles_list: Canonical SMILES strings.
        output_pkl: Target pickle path for `{smiles: embedding_vector}`.
        model_dir: Local MolT5 model directory path.
    """
    # Build tokenizer/encoder once for the whole batch.
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5EncoderModel.from_pretrained(model_dir)
    # Count input entries and prepare output map.
    N_smiles = len(smiles_list)
    smiles_reprs = {}
    # Special handling when all elements are the same SMILES string.
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        # Mean-pool token embeddings (exclude final special token row).
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        # Encode each molecule independently and cache embedding by SMILES key.
        for smile in tqdm(smiles_list):
            input_ids = tokenizer(smile, return_tensors="pt").input_ids
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            smiles_reprs[smile] = embed
            final_values.append(embed.reshape(1, -1))

    # Ensure output directory exists before writing pickle.
    output_dir = join(output_dir, "SMILES")
    os.makedirs(output_dir, exist_ok=True)

    # Persist dictionary for training/inference pipeline reuse.
    with open(join(output_dir, "SMILES_embedding_molT5.pkl"), 'wb') as handle:
        pickle.dump(smiles_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    """Parse command-line arguments for MolT5 embedding generation."""
    parser = argparse.ArgumentParser(description='Generate MolT5 SMILES embeddings from a CSV file.')
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/path/input_data.csv',
        help='Input CSV path containing SMILES strings.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/path/embedding_path/',
        help='Output pickle file path, e.g. /path/.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/path/molt5-base-smiles2caption',
        help='Local MolT5 model directory path.'
    )
    parser.add_argument(
        '--smiles_col',
        type=str,
        default='Smiles',
        help='CSV column name that stores SMILES strings.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    # CLI entry point: read SMILES, canonicalize/deduplicate, and export embeddings.
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    smiles_list = df[args.smiles_col].tolist()[:100]
    smiles_list = [normalize_smiles(x) for x in smiles_list if isinstance(x, str)]
    smiles_list = sorted(list(set(smiles_list)))
    get_molT5_embed(smiles_list, args.output_dir, args.model_dir)
