import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from Github.Model.DelKcat_substrate import DelKcat_gate


# Silence RDKit warnings for cleaner prediction logs.
RDLogger.DisableLog('rdApp.*')


def normalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string.

    Returns:
        Canonical SMILES, or empty string when parsing fails.
    """
    normalized = ''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            normalized = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ''
    return normalized


def preprocess_sequence(seq: str, max_len: int = 1024) -> str:
    """
    Convert an amino-acid sequence into ProtT5 token format.

    - Truncate long sequences to first 512 + last 512 residues.
    - Replace uncommon symbols U/Z/O/B with X.
    - Insert spaces between residues.
    """
    if len(seq) > max_len:
        seq = seq[:512] + seq[-512:]
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def pad_and_mask(emb: torch.Tensor, max_len: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad token embeddings to fixed length and build boolean token mask.
    """
    seq_len, hidden_dim = emb.shape
    if seq_len < max_len:
        padded = torch.zeros((max_len, hidden_dim), dtype=torch.float32)
        padded[:seq_len] = emb
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = True
        return padded, mask
    return emb[:max_len].to(torch.float32), torch.ones(max_len, dtype=torch.bool)


def build_protein_feature_dict(
    sequences: List[str],
    protein_model_dir: str,
    device: torch.device
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build in-memory ProtT5 features for unique protein sequences.

    Dict value:
        (seq_emb, seq_emb_pad, seq_mask)
    """
    tokenizer = T5Tokenizer.from_pretrained(protein_model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(protein_model_dir).to(device)
    model.eval()

    prot_feat: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for seq in tqdm(sorted(set(sequences)), desc='Protein embedding'):
        seq_text = preprocess_sequence(seq)
        ids = tokenizer.batch_encode_plus([seq_text], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Match training preprocessing: remove final special token row.
        seq_emb = hidden.squeeze(0).detach().cpu().to(torch.float32)[:-1, :]
        seq_emb_pad, seq_mask = pad_and_mask(seq_emb, 1024)
        prot_feat[seq] = (seq_emb, seq_emb_pad, seq_mask)

    return prot_feat


def build_smiles_feature_dict(
    smiles_values: List[str],
    substrate_model_dir: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Build in-memory MolT5 mean-pooled embeddings for unique SMILES.
    """
    tokenizer = T5Tokenizer.from_pretrained(substrate_model_dir)
    model = T5EncoderModel.from_pretrained(substrate_model_dir).to(device)
    model.eval()

    smiles_feat: Dict[str, torch.Tensor] = {}
    for smi in tqdm(sorted(set(smiles_values)), desc='SMILES embedding'):
        input_ids = tokenizer(smi, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            last_hidden = model(input_ids=input_ids).last_hidden_state
        embed = torch.mean(last_hidden[0][:-1, :], dim=0).detach().cpu().to(torch.float32)
        smiles_feat[smi] = embed

    return smiles_feat


class InferenceDataset(Dataset):
    """
    Minimal dataset wrapper for already-constructed PyG Data objects.
    """
    def __init__(self, samples: List[Data]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        return self.samples[idx]


def collate_graph_batch(data_list: List[Data]) -> Dict[str, Batch]:
    """
    Build model-compatible batch dict from graph samples.
    """
    return {'dlkcat_features': Batch.from_data_list(data_list)}


def build_samples(
    df: pd.DataFrame,
    protein_features: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    smiles_features: Dict[str, torch.Tensor],
    sequence_col: str,
    smiles1_col: str,
    smiles2_col: str
) -> Tuple[List[Data], int]:
    """
    Convert input dataframe rows into model-ready PyG `Data` samples.

    Returns:
        samples: valid rows converted to Data objects
        skipped: number of dropped invalid rows
    """
    samples: List[Data] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        seq = row[sequence_col]
        smi_1_raw = row[smiles1_col]
        smi_2_raw = row[smiles2_col]

        if not isinstance(seq, str) or not isinstance(smi_1_raw, str) or not isinstance(smi_2_raw, str):
            skipped += 1
            continue

        smi_1 = normalize_smiles(smi_1_raw)
        smi_2 = normalize_smiles(smi_2_raw)
        if smi_1 == '' or smi_2 == '':
            skipped += 1
            continue
        if seq not in protein_features or smi_1 not in smiles_features or smi_2 not in smiles_features:
            skipped += 1
            continue

        mol_1 = AllChem.MolFromSmiles(smi_1)
        mol_2 = AllChem.MolFromSmiles(smi_2)
        if mol_1 is None or mol_2 is None:
            skipped += 1
            continue

        fp_1 = MACCSkeys.GenMACCSKeys(mol_1).ToList()
        fp_2 = MACCSkeys.GenMACCSKeys(mol_2).ToList()

        seq_emb, seq_emb_pad, seq_mask = protein_features[seq]
        mean_seq = torch.mean(seq_emb, dim=0, keepdim=True)

        samples.append(
            Data(
                row_id=torch.tensor([int(row_idx)], dtype=torch.long),
                kcat_gt=torch.tensor(0.0, dtype=torch.float32),
                seq=seq,
                smiles_1=smi_1,
                smiles_2=smi_2,
                seq_emb=seq_emb_pad,
                mask=seq_mask,
                mean_seq=mean_seq,
                smiles_emb_1=torch.FloatTensor(fp_1).unsqueeze(0),
                smiles_emb_2=torch.FloatTensor(fp_2).unsqueeze(0),
                smiles_emb_molt5_1=smiles_features[smi_1].unsqueeze(0),
                smiles_emb_molt5_2=smiles_features[smi_2].unsqueeze(0)
            )
        )

    return samples, skipped


def parse_args():
    """
    Parse CLI arguments for substrate-difference prediction.
    """
    parser = argparse.ArgumentParser(
        description='Predict delta-kcat for substrate pairs without saving intermediate embeddings.'
    )
    parser.add_argument('--input_csv', type=str, default='/path/input_path.csv',
                        help='Input CSV file for prediction.')
    parser.add_argument('--protein_model_dir', type=str, default='/path/prot_t5_xl_uniref50',
                        help='Local ProtT5 model directory.')
    parser.add_argument('--substrate_model_dir', type=str, default='/path/molt5-base-smiles2caption',
                        help='Local MolT5 model directory.')
    parser.add_argument('--deltakcat_model_path', type=str,
                        default='/path/model.pth',
                        help='Pretrained DeltaKcat .pth file.')
    parser.add_argument('--output_csv', type=str,
                        default='/path/output_path.csv',
                        help='Output CSV path with predictions.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Inference device, e.g. cuda:0 or cpu.')
    parser.add_argument('--batch_size', type=int, default=32, help='Prediction batch size.')
    parser.add_argument('--sequence_col', type=str, default='Sequence', help='Protein sequence column.')
    parser.add_argument('--smiles1_col', type=str, default='Smiles_1', help='First substrate SMILES column.')
    parser.add_argument('--smiles2_col', type=str, default='Smiles_2', help='Second substrate SMILES column.')
    parser.add_argument('--pred_col', type=str, default='predict kcat', help='Prediction column name in output CSV.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Model hidden dim used during training.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Model dropout used during training.')
    return parser.parse_args()


def main():
    """
    End-to-end inference workflow:
    1) Read input rows.
    2) Generate ProtT5/MolT5 embeddings in memory.
    3) Construct model features.
    4) Load pretrained substrate DeltaKcat model and predict.
    5) Save predictions to output CSV.
    """
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv(args.input_csv)
    required_cols = [args.sequence_col, args.smiles1_col, args.smiles2_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    seq_values = [x for x in df[args.sequence_col].tolist() if isinstance(x, str)]
    smi_values = []
    for col in [args.smiles1_col, args.smiles2_col]:
        for x in df[col].tolist():
            if isinstance(x, str):
                norm = normalize_smiles(x)
                if norm != '':
                    smi_values.append(norm)

    if not seq_values:
        raise ValueError('No valid protein sequences found in input CSV.')
    if not smi_values:
        raise ValueError('No valid SMILES found in input CSV.')

    print(f'Unique protein sequences: {len(set(seq_values))}')
    print(f'Unique canonical SMILES: {len(set(smi_values))}')

    # Online embedding generation (not saved to disk).
    protein_features = build_protein_feature_dict(seq_values, args.protein_model_dir, device)
    smiles_features = build_smiles_feature_dict(smi_values, args.substrate_model_dir, device)

    samples, skipped = build_samples(
        df=df,
        protein_features=protein_features,
        smiles_features=smiles_features,
        sequence_col=args.sequence_col,
        smiles1_col=args.smiles1_col,
        smiles2_col=args.smiles2_col
    )
    if len(samples) == 0:
        raise ValueError('No valid rows left for prediction after preprocessing.')

    loader = DataLoader(
        InferenceDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graph_batch
    )

    # Build model with the same constructor behavior as train_derta_substrate.
    model = DelKcat_gate(args.hidden_dim, args.dropout, device).to(device)
    ckpt = torch.load(args.deltakcat_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictions = [None] * len(df)
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            batch['dlkcat_features'] = batch['dlkcat_features'].to(device)
            pred = model(batch).detach().cpu().tolist()
            row_ids = batch['dlkcat_features'].row_id.view(-1).detach().cpu().tolist()
            for rid, value in zip(row_ids, pred):
                predictions[rid] = float(value)

    df[args.pred_col] = predictions
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    predicted_count = sum(v is not None for v in predictions)
    print(f'Prediction complete. total={len(df)}, predicted={predicted_count}, skipped={skipped}')
    print(f'Saved: {args.output_csv}')


if __name__ == '__main__':
    # Enable TF32 acceleration for supported GPUs.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
