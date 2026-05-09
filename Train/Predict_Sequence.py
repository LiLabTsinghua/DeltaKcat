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
from Model.DelKcat_sequence import DelKcat_gate


# Disable verbose RDKit logs to keep terminal output clean.
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


def pad_and_mask(emb: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad token embeddings to a fixed length and build a boolean valid-token mask.
    """
    seq_len = emb.shape[0]
    hidden = emb.shape[1]
    if seq_len < max_len:
        padded = torch.zeros((max_len, hidden), dtype=torch.float32)
        padded[:seq_len] = emb
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = True
        return padded, mask
    padded = emb[:max_len].to(torch.float32)
    mask = torch.ones(max_len, dtype=torch.bool)
    return padded, mask


def preprocess_sequence(seq: str, max_len: int = 1024) -> str:
    """
    Convert one amino-acid sequence to ProtT5 token format.

    - Truncate very long sequences by keeping 512 tokens from each terminus.
    - Replace uncommon amino acids (U/Z/O/B) with X.
    - Insert whitespace between amino acids for ProtT5 tokenizer.
    """
    if len(seq) > max_len:
        seq = seq[:512] + seq[-512:]
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def build_protein_feature_dict(
    sequences: List[str],
    protein_model_dir: str,
    device: torch.device
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Encode unique protein sequences with ProtT5 and build in-memory feature dict.

    Dict value format:
        sequence -> (seq_emb, seq_emb_pad, seq_mask)
    """
    tokenizer = T5Tokenizer.from_pretrained(protein_model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(protein_model_dir).to(device)
    model.eval()

    feature_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for seq in tqdm(sorted(set(sequences)), desc='Protein embedding'):
        seq_tokenized = preprocess_sequence(seq)
        ids = tokenizer.batch_encode_plus([seq_tokenized], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Remove final special token row to keep behavior aligned with training preprocessing.
        seq_emb = embedding.squeeze(0).detach().cpu().to(torch.float32)[:-1, :]
        seq_emb_pad, seq_mask = pad_and_mask(seq_emb, 1024)
        feature_dict[seq] = (seq_emb, seq_emb_pad, seq_mask)
    return feature_dict


def build_smiles_feature_dict(
    smiles_list: List[str],
    substrate_model_dir: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Encode unique canonical SMILES with MolT5 and build in-memory feature dict.

    Dict value format:
        canonical_smiles -> mean pooled MolT5 embedding (shape: [768])
    """
    tokenizer = T5Tokenizer.from_pretrained(substrate_model_dir)
    model = T5EncoderModel.from_pretrained(substrate_model_dir).to(device)
    model.eval()

    feature_dict: Dict[str, torch.Tensor] = {}
    for smi in tqdm(sorted(set(smiles_list)), desc='SMILES embedding'):
        input_ids = tokenizer(smi, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            last_hidden_states = model(input_ids=input_ids).last_hidden_state
        embed = torch.mean(last_hidden_states[0][:-1, :], dim=0).detach().cpu().to(torch.float32)
        feature_dict[smi] = embed
    return feature_dict


class InferenceDataset(Dataset):
    """
    Lightweight dataset that stores already-constructed PyG Data samples.
    """
    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


def collate_graph_batch(data_list: List[Data]) -> Dict[str, Batch]:
    """
    Convert a list of PyG Data objects into the model's expected batch dict.
    """
    return {'dlkcat_features': Batch.from_data_list(data_list)}


def build_inference_graphs(
    df: pd.DataFrame,
    protein_features: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    smiles_features: Dict[str, torch.Tensor],
    sequence1_col: str,
    sequence2_col: str,
    smiles_col: str
) -> Tuple[List[Data], int]:
    """
    Transform dataframe rows into model-ready graph samples.

    Returns:
        data_list: valid samples for prediction
        skipped_rows: number of rows skipped due to invalid/missing fields
    """
    data_list: List[Data] = []
    skipped_rows = 0

    for row_idx, row in df.iterrows():
        seq_1 = row[sequence1_col]
        seq_2 = row[sequence2_col]
        raw_smiles = row[smiles_col]

        if not isinstance(seq_1, str) or not isinstance(seq_2, str) or not isinstance(raw_smiles, str):
            skipped_rows += 1
            continue

        smiles = normalize_smiles(raw_smiles)
        if smiles == '' or smiles not in smiles_features:
            skipped_rows += 1
            continue
        if seq_1 not in protein_features or seq_2 not in protein_features:
            skipped_rows += 1
            continue

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            skipped_rows += 1
            continue

        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
        smiles_emb_molt5 = smiles_features[smiles].unsqueeze(0)

        seq_emb_1, seq_emb_pad_1, seq_mask_1 = protein_features[seq_1]
        seq_emb_2, seq_emb_pad_2, seq_mask_2 = protein_features[seq_2]
        mean_seq_1 = torch.mean(seq_emb_1, dim=0, keepdim=True)
        mean_seq_2 = torch.mean(seq_emb_2, dim=0, keepdim=True)

        data_list.append(
            Data(
                row_id=torch.tensor([int(row_idx)], dtype=torch.long),
                smiles_emb=torch.FloatTensor(fp).unsqueeze(0),
                seq_emb_1=seq_emb_pad_1,
                mask_1=seq_mask_1,
                kcat_gt=torch.tensor(0.0, dtype=torch.float32),
                smiles=smiles,
                seq_1=seq_1,
                mean_seq_1=mean_seq_1,
                smiles_emb_molt5=smiles_emb_molt5,
                seq_2=seq_2,
                mean_seq_2=mean_seq_2,
                seq_emb_2=seq_emb_pad_2,
                mask_2=seq_mask_2
            )
        )

    return data_list, skipped_rows


def parse_args():
    """
    Parse command-line arguments for end-to-end DeltaKcat prediction.
    """
    parser = argparse.ArgumentParser(
        description='Predict delta-kcat directly from input CSV without saving intermediate embeddings.'
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction.')
    parser.add_argument('--sequence1_col', type=str, default='Sequence_1', help='Sequence 1 column name.')
    parser.add_argument('--sequence2_col', type=str, default='Sequence_2', help='Sequence 2 column name.')
    parser.add_argument('--smiles_col', type=str, default='Smiles', help='SMILES column name.')
    parser.add_argument('--pred_col', type=str, default='predict kcat', help='Prediction column name in output CSV.')
    return parser.parse_args()


def main():
    """
    End-to-end prediction pipeline:
    1) Read user CSV.
    2) Build protein and molecule embeddings in memory.
    3) Build model input graphs.
    4) Load pretrained DeltaKcat checkpoint and run inference.
    5) Write predictions to output CSV.
    """
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv(args.input_csv)
    required_cols = [args.sequence1_col, args.sequence2_col, args.smiles_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns in input CSV: {missing_cols}')

    # Collect all unique sequences and canonical SMILES before embedding.
    seq_values = list(df[args.sequence1_col].dropna().astype(str).tolist()) + \
                 list(df[args.sequence2_col].dropna().astype(str).tolist())
    canonical_smiles = []
    for smi in df[args.smiles_col].dropna().astype(str).tolist():
        norm = normalize_smiles(smi)
        if norm != '':
            canonical_smiles.append(norm)

    if len(seq_values) == 0:
        raise ValueError('No valid sequences found in input CSV.')
    if len(canonical_smiles) == 0:
        raise ValueError('No valid SMILES found in input CSV.')

    print(f'Unique protein sequences: {len(set(seq_values))}')
    print(f'Unique canonical SMILES: {len(set(canonical_smiles))}')

    # Generate embeddings on-the-fly; no intermediate embedding files are saved.
    protein_features = build_protein_feature_dict(seq_values, args.protein_model_dir, device)
    smiles_features = build_smiles_feature_dict(canonical_smiles, args.substrate_model_dir, device)

    data_list, skipped_rows = build_inference_graphs(
        df=df,
        protein_features=protein_features,
        smiles_features=smiles_features,
        sequence1_col=args.sequence1_col,
        sequence2_col=args.sequence2_col,
        smiles_col=args.smiles_col
    )
    if len(data_list) == 0:
        raise ValueError('No valid rows available for prediction after preprocessing.')

    dataset = InferenceDataset(data_list)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graph_batch
    )

    # Build model with the same constructor signature used by project training scripts.
    model = DelKcat_gate(167, 512, 1280, 1, 0, 10, device).to(device)
    ckpt = torch.load(args.deltakcat_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    # Accept checkpoints saved by DataParallel (keys prefixed with "module.").
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
            for rid, p in zip(row_ids, pred):
                predictions[rid] = float(p)

    # Write predictions back to original dataframe order.
    df[args.pred_col] = predictions
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    predicted_count = sum(x is not None for x in predictions)
    print(f'Prediction complete. Total rows: {len(df)}, predicted: {predicted_count}, skipped: {skipped_rows}')
    print(f'Output saved to: {args.output_csv}')


if __name__ == '__main__':
    # Keep TF32 acceleration enabled when running on compatible GPUs.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
