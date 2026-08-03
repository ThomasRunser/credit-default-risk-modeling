import os
import pickle
import sys

import numpy as np
import pandas as pd

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)


def convert_types(df):
    print(f'Original size of data: {return_size(df)} gb.')
    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    print(f'New size of data: {return_size(df)} gb.')
    return df


def csv_to_pkl(override=False):
    path = "./data/"
    
    csv_files = ['application_test.csv',
                 'application_train.csv',
                 'bureau.csv',
                 'bureau_balance.csv',
                 'credit_card_balance.csv',
                 'installments_payments.csv',
                 'POS_CASH_balance.csv',
                 'previous_application.csv']

    for f in csv_files:
        isExist = os.path.exists(path+os.path.splitext(f)[0]+'.pkl')

        if(not(isExist) or override):
            print('-'*20)
            print(os.path.splitext(f)[0])
            print('-'*20)
            _df = pd.read_csv(path+f, 
                              sep = ',', 
                              #quoting=csv.QUOTE_NONE,
                              on_bad_lines = 'warn',
                              na_values=['XNA', 'XAP'],
                              engine='python') 
            

            print('dataset shape', _df.shape)
            print('duplicated:', _df.duplicated().sum())
            
            days_cols = [col for col in _df.columns if 'DAYS' in col]
            print('Value 365243 in', days_cols, 'will be replaced by nan')
            _df[days_cols] = _df[days_cols].replace(365243, np.nan)

            _df = convert_types(_df)
            
            #Save dataframe as pickle file
            with open('./data/{}.pkl'.format(os.path.splitext(f)[0]), 'wb') as pickle_file:
                pickle.dump(_df, pickle_file)


def reduce_memory(df):
    """Reduce memory usage without fragmenting the DataFrame."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    print(
        f"Initial df memory usage is {start_mem:.2f} MB "
        f"for {len(df.columns)} columns"
    )

    dtype_map = {}

    for col in df.columns:
        col_type = df[col].dtype

        if not pd.api.types.is_numeric_dtype(col_type):
            continue

        cmin = df[col].min()
        cmax = df[col].max()

        # Keep columns containing NaN as floats
        has_nan = df[col].isna().any()

        if pd.api.types.is_integer_dtype(col_type) and not has_nan:
            if cmin >= np.iinfo(np.int8).min and cmax <= np.iinfo(np.int8).max:
                dtype_map[col] = np.int8
            elif cmin >= np.iinfo(np.int16).min and cmax <= np.iinfo(np.int16).max:
                dtype_map[col] = np.int16
            elif cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
                dtype_map[col] = np.int32
            else:
                dtype_map[col] = np.int64
        else:
            if cmin >= np.finfo(np.float32).min and cmax <= np.finfo(np.float32).max:
                dtype_map[col] = np.float32
            else:
                dtype_map[col] = np.float64

    # Convert every column together instead of inserting blocks repeatedly
    df = df.astype(dtype_map).copy()

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem

    print(
        f"Final memory usage is {end_mem:.2f} MB "
        f"- decreased by {reduction:.1f}%"
    )

    return df
