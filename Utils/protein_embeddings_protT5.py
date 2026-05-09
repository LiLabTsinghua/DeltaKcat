import argparse
import os
import re
from os.path import join
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from rdkit import Chem
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def create_file(path):
    """
    Create a directory if missing; otherwise clear files inside it.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        all_files = os.listdir(path)
        for file in all_files:
            file_path = join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


def create_fasta_file(sequences, filename):
    """
    Write raw sequences to a FASTA file, using row index as record id.
    """
    with open(filename, "w") as ofile:
        for k, seq in enumerate(sequences):
            ofile.write(">" + str(k) + "\n" + seq + "\n")


def build_sequence_index_map(fasta_file):
    """
    Build a map: sequence string -> integer id in FASTA order.
    """
    sequence_map = {str(record.seq): idx for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta"))}
    return sequence_map


def protein_embedding_prottrans(sequences, output_path, model_path, device='cuda:0'):
    """
    Generate per-sequence ProtT5 embeddings and save temporary `.pt` files.

    Outputs:
    - {output_path}/sequences_prottrans.fasta
    - {output_path}/Protein/temp/{protein_id}.pt
    """
    fasta_file = join(output_path, "sequences_prottrans.fasta")
    protein_file = join(output_path, "Protein")
    create_fasta_file(sequences, fasta_file)
    sequence_map = build_sequence_index_map(fasta_file)
    create_file(protein_file)

    # Load tokenizer/model from user-provided local model path.
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Transferred model to GPU: {device}")
    else:
        device = 'cpu'
        print("CUDA not available. Running on CPU.")

    label_list = [str(sequence_map.get(x, -1)) for x in sequences]
    output_dir = join(output_path, "Protein", "temp")
    create_file(output_dir)

    # Truncate sequences longer than 1024 by keeping both termini.
    for i in range(len(sequences)):
        if len(sequences[i]) > 1024:
            sequences[i] = sequences[i][:512] + sequences[i][-512:]

    # Replace uncommon amino acids and convert to whitespace-separated tokens.
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    for i in tqdm(range(len(sequences))):
        print('For sequence ', str(i + 1))
        output_file = join(output_dir, label_list[i] + ".pt")
        sequences_example_i = sequences[i]
        sequences_example_i = [re.sub(r"[UZOB]", "X", sequences_example_i)]
        ids = tokenizer.batch_encode_plus(sequences_example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        torch.save(embedding, output_file)


def save_prottrans_features(seq_list, output_path):
    """
    Read temporary per-sequence embeddings, pad/mask to fixed length,
    then save one dictionary file for training code.

    Output:
    - {output_path}/prottrans_feature_dict.pt
    """
    seq_list = [x for x in seq_list if isinstance(x, str)]
    seq_path = os.path.join(output_path, 'Protein', 'temp', '{}.pt')
    sequence_map = build_sequence_index_map(os.path.join(output_path, 'sequences_prottrans.fasta'))
    esm_dict = {}

    for seq in tqdm(seq_list):
        pro_id = sequence_map.get(seq, -1)
        seq_emb = torch.load(seq_path.format(pro_id), weights_only=False)
        seq_emb = torch.tensor(seq_emb).squeeze(0)
        seq_emb = seq_emb[:seq_emb.shape[0] - 1, :]
        seq_emb_pad, seq_mask = pad_and_mask(seq_emb, 1024)
        esm_dict[pro_id] = [seq_emb, seq_emb_pad, seq_mask]

    torch.save(esm_dict, os.path.join(output_path, 'prottrans_feature_dict.pt'))


def pad_and_mask(emb, max_len):
    """
    Pad sequence embeddings to `max_len` and build a valid-token mask.
    """
    if len(emb) < max_len:
        padded_emb = np.pad(emb, ((0, max_len - len(emb)), (0, 0)), mode='constant', constant_values=0)
        mask = np.concatenate([np.ones(len(emb)), np.zeros(max_len - len(emb))])
    else:
        padded_emb = emb[:max_len]
        mask = np.ones(max_len)
    return torch.FloatTensor(padded_emb), torch.BoolTensor(mask)


def parse_args():
    """
    Parse command-line arguments for ProtT5 embedding generation.
    """
    parser = argparse.ArgumentParser(description='Generate ProtT5 protein embeddings from a CSV sequence column.')
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/path/input_data.csv',
        help='Input CSV file path that contains protein sequences.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/path/output_dir/',
        help='Output directory for FASTA, temporary embeddings, and final feature dict.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/path/prot_t5_xl_uniref50',
        help='Local ProtT5 model directory path for tokenizer/model loading.'
    )
    parser.add_argument(
        '--sequence_col',
        type=str,
        default='Sequence',
        help='Column name in CSV that stores amino-acid sequences.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Torch device string, e.g. cuda:0 or cpu.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Ensure output root exists before creating child folders/files.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load unique sequence list from user-provided CSV/column.
    df = pd.read_csv(args.input_csv)
    seq_list = sorted(list(set(df[args.sequence_col].tolist())))
    seq_list = [x for x in seq_list if isinstance(x, str)]

    # Step 1: generate per-sequence temporary embeddings.
    protein_embedding_prottrans(
        sequences=seq_list,
        output_path=args.output_dir,
        model_path=args.model_dir,
        device=args.device
    )
    # Step 2: aggregate temporary embeddings into one training dictionary.
    save_prottrans_features(seq_list, args.output_dir)
