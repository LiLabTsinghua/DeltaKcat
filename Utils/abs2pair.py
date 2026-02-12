import math

import pandas as pd
from tqdm import tqdm


def abs2pair_sub(result_path, test_path, output_path):
    unikp_result = pd.read_csv(result_path)
    pair_test = pd.read_csv(test_path)
    pair_test = pair_test[(pair_test['Value_1'] != 0) & (pair_test['Value_2'] != 0)]

    for index, row in tqdm(pair_test.iterrows()):
        seq = row['Sequence']
        pubmed = row['PubmedID']
        smiles_1 = row['Smiles_1']
        smiles_2 = row['Smiles_2']
        try:
            mask1 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq) & (
                        unikp_result['Smiles'] == smiles_1)
            mask2 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq) & (
                        unikp_result['Smiles'] == smiles_2)
            s1 = unikp_result.loc[mask1, 'predict kcat']
            s2 = unikp_result.loc[mask2, 'predict kcat']
            if (not s1.empty) and (not s2.empty):
                unikp_result_1 = s1.iloc[0]
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - unikp_result_2
            elif (not s1.empty) and s2.empty:
                unikp_result_1 = s1.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - math.log2(row['Value_2'])
            elif s1.empty and (not s2.empty):
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = math.log2(row['Value_1']) - unikp_result_2
            else:
                print('Error')

            # unikp_result_1 = unikp_result[(unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq)
            #                               & (unikp_result['Smiles'] == smiles_1)]['predict kcat'].iloc[0]
            # unikp_result_2 = unikp_result[(unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq)
            #                               & (unikp_result['Smiles'] == smiles_2)]['predict kcat'].iloc[0]
            # pair_test.loc[index, 'predict kcat'] = unikp_result_1 - unikp_result_2
        except:
            continue
    pair_test.to_csv(output_path, index=False)


def abs2pair_seq(result_path, test_path, output_path):
    unikp_result = pd.read_csv(result_path)
    pair_test = pd.read_csv(test_path)
    pair_test = pair_test[(pair_test['Value_1'] != 0) & (pair_test['Value_2'] != 0)]

    for index, row in tqdm(pair_test.iterrows()):
        seq_1 = row['Sequence_1']
        seq_2 = row['Sequence_2']
        pubmed = row['PubmedID']
        smiles = row['Smiles']
        try:
            mask1 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq_1) & (
                        unikp_result['Smiles'] == smiles)
            mask2 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq_2) & (
                        unikp_result['Smiles'] == smiles)
            s1 = unikp_result.loc[mask1, 'predict kcat']
            s2 = unikp_result.loc[mask2, 'predict kcat']
            if (not s1.empty) and (not s2.empty):
                unikp_result_1 = s1.iloc[0]
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - unikp_result_2
            elif (not s1.empty) and s2.empty:
                unikp_result_1 = s1.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - math.log2(row['Value_2'])
            elif s1.empty and (not s2.empty):
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = math.log2(row['Value_1']) - unikp_result_2
            else:
                print('Error')

            # unikp_result_1 = unikp_result[(unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq)
            #                               & (unikp_result['Smiles'] == smiles_1)]['predict kcat'].iloc[0]
            # unikp_result_2 = unikp_result[(unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq)
            #                               & (unikp_result['Smiles'] == smiles_2)]['predict kcat'].iloc[0]
            # pair_test.loc[index, 'predict kcat'] = unikp_result_1 - unikp_result_2
        except:
            continue
    pair_test.to_csv(output_path, index=False)


def abs2pair_seq_adk(result_path, test_path, output_path):
    unikp_result = pd.read_csv(result_path)
    pair_test = pd.read_csv(test_path)
    pair_test = pair_test[(pair_test['Value_1'] != 0) & (pair_test['Value_2'] != 0)]
    unikp_result['PubmedID'] = 1
    pair_test['PubmedID'] = 1

    for index, row in tqdm(pair_test.iterrows()):
        seq_1 = row['Sequence_1']
        seq_2 = row['Sequence_2']
        pubmed = row['PubmedID']
        smiles = row['Smiles']
        try:
            mask1 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq_1)
            mask2 = (unikp_result['PubmedID'] == pubmed) & (unikp_result['Sequence'] == seq_2)
            s1 = unikp_result.loc[mask1, 'predict kcat']
            s2 = unikp_result.loc[mask2, 'predict kcat']
            if (not s1.empty) and (not s2.empty):
                unikp_result_1 = s1.iloc[0]
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - unikp_result_2
            elif (not s1.empty) and s2.empty:
                unikp_result_1 = s1.iloc[0]
                pair_test.loc[index, 'predict kcat'] = unikp_result_1 - math.log2(row['Value_2'])
                print('Error')
            elif s1.empty and (not s2.empty):
                unikp_result_2 = s2.iloc[0]
                pair_test.loc[index, 'predict kcat'] = math.log2(row['Value_1']) - unikp_result_2
                print('Error')
            else:
                print('Error')
        except:
            continue
    pair_test.to_csv(output_path, index=False)

if __name__ == '__main__':

    abs2pair_seq_adk(result_path='../Result/CataPro_result/ADK_kcat_catapro_seq.csv',
                 test_path='../Data/ADK/adk_pair_seq_kcat2.csv',
                 output_path='../Result/CataPro_result/abs2pair/ADK_kcat_catapro_seq_abs2pair.csv')