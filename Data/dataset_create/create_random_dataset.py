import math

import pandas as pd
from sklearn.model_selection import train_test_split


def create_random_dataset_sub():
    df_all = pd.read_csv('../Data/kkm/kkm_diff_substrate.csv')
    df_all = df_all[(df_all['Value']>0) & (df_all['Value']<1000000)]
    train_set, test_set = train_test_split(df_all, test_size=0.18, random_state=42)

    train_pair_set = pd.merge(
        train_set,
        train_set,
        on=['Sequence', 'PubmedID'],
        suffixes=('_1', '_2')
    )
    test_pair_set = pd.merge(
        df_all,
        test_set,
        on=['Sequence', 'PubmedID'],
        suffixes=('_1', '_2')
    )
    final_pairs_df_train = train_pair_set[train_pair_set['Smiles_1'] != train_pair_set['Smiles_2']].copy()
    final_pairs_df_train = final_pairs_df_train[final_pairs_df_train['Smiles_1'] < final_pairs_df_train['Smiles_2']].copy()
    final_pairs_df_train = final_pairs_df_train[(final_pairs_df_train['Value_1']>0) & (final_pairs_df_train['Value_2']>0)].copy()
    final_pairs_df_train['kcat_ratio'] = final_pairs_df_train['Value_1'] / final_pairs_df_train['Value_2']
    final_pairs_df_train['log2_delta_kcat'] = final_pairs_df_train['kcat_ratio'].apply(math.log2)


    final_pairs_df_test = test_pair_set[test_pair_set['Smiles_1'] != test_pair_set['Smiles_2']].copy()
    final_pairs_df_test = final_pairs_df_test[
        final_pairs_df_test['Smiles_1'] < final_pairs_df_test['Smiles_2']].copy()
    final_pairs_df_test = final_pairs_df_test[
        (final_pairs_df_test['Value_1'] > 0) & (final_pairs_df_test['Value_2'] > 0)].copy()
    final_pairs_df_test['kcat_ratio'] = final_pairs_df_test['Value_1'] / final_pairs_df_test['Value_2']
    final_pairs_df_test['log2_delta_kcat'] = final_pairs_df_test['kcat_ratio'].apply(math.log2)

    print(len(final_pairs_df_test))
    print(len(final_pairs_df_train))

    train_set.to_csv('../Data/kkm/Train_kkm_8_2_substrate_random.csv', index=False)
    test_set.to_csv('../Data/kkm/Test_kkm_8_2_substrate_random.csv', index=False)

    final_pairs_df_train.to_csv('../Data/kkm_pair_sub/Train_pair_sub_kkm_random.csv', index=False)
    final_pairs_df_test.to_csv('../Data/kkm_pair_sub/Test_pair_sub_kkm_random.csv',
                                index=False)


def create_random_dataset_seq():
    df_all = pd.read_csv('../Data/kkm/kkm_diff_sequence.csv')
    df_all = df_all[df_all['Value'] > 0]
    train_set, test_set = train_test_split(df_all, test_size=0.18, random_state=42)

    train_pair_set = pd.merge(
        train_set,
        train_set,
        on=['Smiles', 'PubmedID'],
        suffixes=('_1', '_2')
    )
    test_pair_set = pd.merge(
        df_all,
        test_set,
        on=['Smiles', 'PubmedID'],
        suffixes=('_1', '_2')
    )
    final_pairs_df_train = train_pair_set[train_pair_set['Sequence_1'] != train_pair_set['Sequence_2']].copy()
    final_pairs_df_train = final_pairs_df_train[final_pairs_df_train['Sequence_1'] < final_pairs_df_train['Sequence_2']].copy()
    final_pairs_df_train = final_pairs_df_train[(final_pairs_df_train['Value_1']>0) & (final_pairs_df_train['Value_2']>0)].copy()
    final_pairs_df_train['kcat_ratio'] = final_pairs_df_train['Value_1'] / final_pairs_df_train['Value_2']
    final_pairs_df_train['log2_delta_kcat'] = final_pairs_df_train['kcat_ratio'].apply(math.log2)


    final_pairs_df_test = test_pair_set[test_pair_set['Sequence_1'] != test_pair_set['Sequence_2']].copy()
    final_pairs_df_test = final_pairs_df_test[
        final_pairs_df_test['Sequence_1'] < final_pairs_df_test['Sequence_2']].copy()
    final_pairs_df_test = final_pairs_df_test[
        (final_pairs_df_test['Value_1'] > 0) & (final_pairs_df_test['Value_2'] > 0)].copy()
    final_pairs_df_test['kcat_ratio'] = final_pairs_df_test['Value_1'] / final_pairs_df_test['Value_2']
    final_pairs_df_test['log2_delta_kcat'] = final_pairs_df_test['kcat_ratio'].apply(math.log2)

    train_set.to_csv('../Data/kkm/Train_kkm_8_2_sequence_random.csv', index=False)
    test_set.to_csv('../Data/kkm/Test_kkm_8_2_sequence_random.csv', index=False)

    print(len(final_pairs_df_test))
    print(len(final_pairs_df_train))

    final_pairs_df_train.to_csv('../Data/kkm_pair_seq/Train_pair_seq_kkm_random.csv', index=False)
    final_pairs_df_test.to_csv('../Data/kkm_pair_seq/Test_pair_seq_kkm_random.csv',
                                index=False)


def create_random_dataset_all():
    df_all = pd.read_csv('../kkm/kkm_diff_all.csv')
    df_all = df_all[df_all['Value'] > 0]
    train_set, test_set = train_test_split(df_all, test_size=0.199, random_state=42)

    train_pair_set = pd.merge(
        train_set,
        train_set,
        on=['PubmedID'],
        suffixes=('_1', '_2')
    )
    test_pair_set = pd.merge(
        df_all,
        test_set,
        on=['PubmedID'],
        suffixes=('_1', '_2')
    )

    mask_not_equal = (train_pair_set['Sequence_1'] != train_pair_set['Sequence_2']) | \
                     (train_pair_set['Smiles_1'] != train_pair_set['Smiles_2'])
    final_pairs_df_train = train_pair_set[mask_not_equal].copy()

    mask_order = (final_pairs_df_train['Sequence_1'] < final_pairs_df_train['Sequence_2']) | \
                 ((final_pairs_df_train['Sequence_1'] == final_pairs_df_train['Sequence_2']) &
                  (final_pairs_df_train['Smiles_1'] < final_pairs_df_train['Smiles_2']))
    final_pairs_df_train = final_pairs_df_train[mask_order].copy()
    final_pairs_df_train = final_pairs_df_train[(final_pairs_df_train['Value_1']>0) & (final_pairs_df_train['Value_2']>0)].copy()
    final_pairs_df_train['kcat_ratio'] = final_pairs_df_train['Value_1'] / final_pairs_df_train['Value_2']
    final_pairs_df_train = final_pairs_df_train[ (final_pairs_df_train['kcat_ratio']>0) & (final_pairs_df_train['kcat_ratio']<100000)]
    final_pairs_df_train['log2_delta_kcat'] = final_pairs_df_train['kcat_ratio'].apply(math.log2)


    mask_not_equal = (test_pair_set['Sequence_1'] != test_pair_set['Sequence_2']) | \
                     (test_pair_set['Smiles_1'] != test_pair_set['Smiles_2'])
    final_pairs_df_test = test_pair_set[mask_not_equal].copy()

    mask_order = (final_pairs_df_test['Sequence_1'] < final_pairs_df_test['Sequence_2']) | \
                 ((final_pairs_df_test['Sequence_1'] == final_pairs_df_test['Sequence_2']) &
                  (final_pairs_df_test['Smiles_1'] < final_pairs_df_test['Smiles_2']))
    final_pairs_df_test = final_pairs_df_test[mask_order].copy()

    final_pairs_df_test = final_pairs_df_test[
        (final_pairs_df_test['Value_1'] > 0) & (final_pairs_df_test['Value_2'] > 0)].copy()
    final_pairs_df_test['kcat_ratio'] = final_pairs_df_test['Value_1'] / final_pairs_df_test['Value_2']
    final_pairs_df_test = final_pairs_df_test[
        (final_pairs_df_test['kcat_ratio'] > 0) & (final_pairs_df_test['kcat_ratio'] < 100000)]
    final_pairs_df_test['log2_delta_kcat'] = final_pairs_df_test['kcat_ratio'].apply(math.log2)

    train_set.to_csv('../kkm/Train_kkm_8_2_all_random.csv', index=False)
    test_set.to_csv('../kkm/Test_kkm_8_2_all_random.csv', index=False)

    print(len(final_pairs_df_test))
    print(len(final_pairs_df_train))

    final_pairs_df_train.to_csv('../kkm_pair_all/Train_pair_all_kkm_random.csv', index=False)
    final_pairs_df_test.to_csv('../kkm_pair_all/Test_pair_all_kkm_random.csv',
                                index=False)

if __name__ == '__main__':
    # create_random_dataset_sub()
    # create_random_dataset_seq()
    create_random_dataset_all()