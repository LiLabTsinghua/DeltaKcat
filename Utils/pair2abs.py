import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import math
from tqdm import tqdm


def pair2abs_sub(train_path, result_path, output_path):
    df_train = pd.read_csv(train_path)
    df_train_smiles_1 = df_train['Smiles_1'].tolist()
    df_train_smiles_2 = df_train['Smiles_2'].tolist()
    df_train_smiles_1.extend(df_train_smiles_2)
    df_train_smiles = sorted(list(set(df_train_smiles_1)))
    df_test = pd.read_csv(result_path)
    df_test = df_test[(df_test['Value_1'] != 0) & (df_test['Value_2'] != 0)]
    df_test = df_test[(df_test['log2_delta_kcat'].notna()) & (df_test['predict kcat'].notna())]
    df_test = df_test[(df_test['Smiles_1'].isin(df_train_smiles)) | (df_test['Smiles_2'].isin(df_train_smiles))]

    columns = ['PubmedID', 'Sequence', 'kcat_ratio', 'log2_delta_kcat', 'predict kcat']
    cols_map_2 = {col: col[:-2] for col in df_test.columns if col.endswith('_2')}
    cols_map_1 = {col: col[:-2] for col in df_test.columns if col.endswith('_1')}
    columns += list(cols_map_2.values())

    df_result = pd.DataFrame(columns=columns)

    for index, row in tqdm(df_test.iterrows()):
        new_row = {}
        new_row['PubmedID'] = row['PubmedID']
        new_row['Sequence'] = row['Sequence']
        new_row['kcat_ratio'] = row['kcat_ratio']
        new_row['log2_delta_kcat'] = row['log2_delta_kcat']
        if row['Smiles_1'] in df_train_smiles:
            new_row['predict kcat'] = math.log2(row['Value_1']) - row['predict kcat']
            for col_old, col_new in cols_map_2.items():
                new_row[col_new] = row[col_old]
        else:
            new_row['predict kcat'] = math.log2(row['Value_2']) + row['predict kcat']
            for col_old, col_new in cols_map_1.items():
                new_row[col_new] = row[col_old]
        df_result.loc[len(df_result)] = new_row
    df_result['predict kcat'] = df_result.groupby(['Smiles', 'Sequence', 'PubmedID'])['predict kcat'].transform('mean')
    df_result = df_result.drop_duplicates(subset=['Smiles', 'Sequence', 'PubmedID'])
    df_result.to_csv(output_path, index=False)


def pair2abs_seq(train_path, result_path, output_path):
    df_train = pd.read_csv(train_path)
    df_train_smiles_1 = df_train['Sequence_1'].tolist()
    df_train_smiles_2 = df_train['Sequence_2'].tolist()
    df_train_smiles_1.extend(df_train_smiles_2)
    df_train_smiles = sorted(list(set(df_train_smiles_1)))
    print(len(df_train_smiles))
    df_test = pd.read_csv(result_path)
    df_test = df_test[(df_test['Value_1'] != 0) & (df_test['Value_2'] != 0)]
    df_test = df_test[(df_test['log2_delta_kcat'].notna()) & (df_test['predict kcat'].notna())]
    df_test = df_test[(df_test['Sequence_1'].isin(df_train_smiles)) | (df_test['Sequence_2'].isin(df_train_smiles))]

    columns = ['PubmedID', 'Smiles', 'kcat_ratio', 'log2_delta_kcat', 'predict kcat']
    cols_map_2 = {col: col[:-2] for col in df_test.columns if col.endswith('_2')}
    cols_map_1 = {col: col[:-2] for col in df_test.columns if col.endswith('_1')}
    columns += list(cols_map_2.values())

    df_result = pd.DataFrame(columns=columns)

    for index, row in tqdm(df_test.iterrows()):
        new_row = {}
        new_row['PubmedID'] = row['PubmedID']
        new_row['Smiles'] = row['Smiles']
        new_row['kcat_ratio'] = row['kcat_ratio']
        new_row['log2_delta_kcat'] = row['log2_delta_kcat']
        if row['Sequence_1'] in df_train_smiles:
            new_row['predict kcat'] = math.log2(row['Value_1']) - row['predict kcat']
            for col_old, col_new in cols_map_2.items():
                new_row[col_new] = row[col_old]
        else:
            new_row['predict kcat'] = math.log2(row['Value_2']) + row['predict kcat']
            for col_old, col_new in cols_map_1.items():
                new_row[col_new] = row[col_old]
        df_result.loc[len(df_result)] = new_row
    df_result['predict kcat'] = df_result.groupby(['Smiles', 'Sequence', 'PubmedID'])['predict kcat'].transform('mean')
    df_result = df_result.drop_duplicates(subset=['Smiles', 'Sequence', 'PubmedID'])
    df_result.to_csv(output_path, index=False)


if __name__ == '__main__':
    pair2abs_sub(train_path='../Data/kcat_pair_sub/Train_pair_sub_kcat_cold.csv',
                 result_path='../Result/kcat/Test_cold_kcat_sub_gate.csv',
                 output_path='../Result/kcat/Test_cold_kcat_sub_gate_pair2abs.csv')

    pair2abs_seq(train_path='../Data/kcat_pair_seq/Train_pair_seq_kcat_cold.csv',
                 result_path='../Result/kcat/Test_cold_kcat_seq_gate.csv',
                 output_path='../Result/kcat/Test_cold_kcat_seq_gate_pair2abs.csv')
