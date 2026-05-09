# DeltaKcat

## Introduction
DeltaKcat is a pairwise relative learning framework for robust enzyme kinetics prediction. Instead of directly regressing noisy absolute kinetic values, DeltaKcat learns relative differences between enzyme–substrate complexes within consistent experimental contexts. This approach improves generalization to unseen enzymes and substrates and supports the discovery of highly active enzyme candidates.


## 1. Environment setup

DeltaKcat can be installed using the provided `environment.yml` file. We recommend creating an isolated Conda environment to ensure consistent package versions.

```powershell
conda env create -f environment.yml
conda activate DeltaKcat
```

### 1.1 Dependencies
We used the following Python packages for core development. We tested on `Python 3.11`.

| name            | version |
|-----------------|---------|
| numpy           | 1.24.4  |
| pandas          | 2.0.3   |
| networkx        | 3.5     |
| tqdm            | 4.66.5  |
| torch           | 2.2.0   |
| torch-geometric | 2.7.0   |
| scipy           | 1.16.2  |
| transformers    | 4.40.0  |
| scikit-learn    | 1.2.0   |
| bio             | 1.8.1   |
| rdkit           |         |
| pickle          |         |
| collections     |         |

## 2. Training Workflow

Training scripts:
- Sequence-delta task: `/Train/train_delta_sequence.py`
- Substrate-delta task: `/Train/train_derta_substrate.py`

Both scripts expect this embedding layout under one `embedding_path`:
- `sequences_prottrans.fasta`
- `prottrans_feature_dict.pt`
- `SMILES/SMILES_embedding_molT5.pkl`

### 2.1 Generate protein embeddings (ProtT5)

Script: `/Utils/protein_embeddings_protT5.py`. ProtT5 model could be download from https://huggingface.co/Rostlab/prot_t5_xl_uniref50.

```powershell
python /Utils/protein_embeddings_protT5.py `
  --input_csv /path/data_file `
  --output_dir /path/embedding_dir `
  --model_dir /path/prot_t5_xl_uniref50 `
  --sequence_col Sequence `
  --device cuda:0
```

Notes:
- Writes `sequences_prottrans.fasta` and `prottrans_feature_dict.pt` into `output_dir`.
- Default `--sequence_col` is `Sequence`.
- If your data has `Sequence_1` and `Sequence_2`, prepare a temporary CSV with one merged `Sequence` column.

### 2.2 Generate SMILES embeddings (MolT5)

Script: `/Utils/smiles_embeddings_molt5.py`. molt5 model could be download from https://huggingface.co/laituan245/molt5-base-smiles2caption.

```powershell
python /Utils/smiles_embeddings_molt5.py `
  --input_csv /path/data_file `
  --output_dir /path/embedding_dir `
  --model_dir /path/molt5-base-smiles2caption `
  --smiles_col Smiles
```

Notes:
- Writes `/path/embedding_dir/SMILES/SMILES_embedding_molT5.pkl`.
- Use the same `output_dir` as protein embedding step.

### 2.3 Train models

#### 2.3.1 Train sequence-delta model

```powershell
python /Train/train_delta_sequence.py `
  --train_path /path/Train_pair_seq.csv `
  --test_path /path/Test_pair_seq.csv `
  --embedding_path /path/embedding_dir `
  --output_path /path/result_dir/ `
  --device 0 `
  --batch_size 64 `
  --epoch 100 `
  --lr 5e-5
```

Input CSV should include at least:
- `Sequence_1`
- `Sequence_2`
- `Smiles`
- `Value_1`
- `Value_2`
- `log2_delta_kcat`

#### 2.3.2 Train substrate-delta model

```powershell
python /Train/train_derta_substrate.py `
  --train_path /path/Train_substrate.csv `
  --test_path /path/Test_substrate.csv `
  --embedding_path /path/embedding_dir `
  --output_path /path/result_dir/ `
  --device 0 `
  --batch_size 64 `
  --epoch 100 `
  --lr 5e-5
```

Input CSV should include at least:
- `Sequence`
- `Smiles_1`
- `Smiles_2`
- `Value_1`
- `Value_2`
- `log2_delta_kcat`

## 3. Prediction Workflow

Download and unzip the [model parameters](https://drive.google.com/file/d/1iDKG1cgIDmzABeKV36R9oL0SrhVSaSB0/view?usp=sharing) under DeltaKcat

### 3.1 Sequence-delta prediction

Script: `/Train/predict_end2end.py`

```powershell
python /Train/predict_end2end.py `
  --input_csv /path/predict_seq.csv `
  --protein_model_dir /path/prot_t5_xl_uniref50 `
  --substrate_model_dir /path/molt5-base-smiles2caption `
  --deltakcat_model_path /path/Delta_seq_model.pth `
  --output_csv /path/predict_seq_out.csv `
  --device cuda:0
```

Default input columns:
- `Sequence_1`
- `Sequence_2`
- `Smiles`

Optional column override:
- `--sequence1_col`
- `--sequence2_col`
- `--smiles_col`

### 3.2 Substrate-delta prediction

Script: `/Train/predict_substrate_end2end.py`

```powershell
python /Train/predict_substrate_end2end.py `
  --input_csv /path/predict_sub.csv `
  --protein_model_dir /path/prot_t5_xl_uniref50 `
  --substrate_model_dir /path/molt5-base-smiles2caption `
  --deltakcat_model_path /path/Delta_sub_model.pth `
  --output_csv /path/predict_sub_out.csv `
  --device cuda:0
```

Default input columns:
- `Sequence`
- `Smiles_1`
- `Smiles_2`

Optional column override:
- `--sequence_col`
- `--smiles1_col`
- `--smiles2_col`

## 4. Common Issues

### 4.1 `ModuleNotFoundError`
- Confirm you run from project root.
- Run: `$env:PYTHONPATH = (Get-Location).Path`

### 4.2 CUDA not available
- Use `--device cpu`.

### 4.3 Missing prediction values
- Invalid sequences/SMILES are skipped during preprocessing.
- Check script summary logs for skipped row count.


Contact
-------

-   Feiran Li ([@feiranl](https://github.com/feiranl)), Tsinghua University, Shenzhen, China
-   Xiongwen Li ([@xiongwenL](https://github.com/xiongwenL)), Tsinghua University, Shenzhen, China
