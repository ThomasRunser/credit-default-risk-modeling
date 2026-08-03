import pickle
import re

import pandas as pd

from .data_io import reduce_memory
from .related_tables import (
    bureau_bbalance,
    previous_application,
    credit_card_balance,
    installments_payments,
    pos_cash_balance,
)

def get_merged_dataframe(debug=False, load=False):
    if load:
        with open('./data/merged_dataset.pkl', 'rb') as pickle_file:
            df = pickle.load(pickle_file)

        print('*'*20)
        print('LOAD MODE\nLOADING THE PREVIOUS MERGED DATAFRAME')
        print('*'*20)
        print('final merged dataframe shape:', df.shape)
        return df.loc[('train',),:], df.loc[('test',),:]
    
    if debug:
        n_rows = 30000
        print('*'*20)
        print('DEBUG MODE\nONLY THE FIRST', n_rows, 'ROWS ARE LOADED')
        print('*'*20)
    else:
        n_rows = None
    
    print('application train and application test')
    with open('./data/application_prepared.pkl', 'rb') as pickle_file:
        df = pickle.load(pickle_file)

    if debug:
        train = df.loc[('train',), :].iloc[:n_rows, :].copy()
        test = df.loc[('test',), :].iloc[:n_rows, :].copy()
        df = pd.concat([train, test], keys=['train', 'test'])

    print('application dataframe shape:', df.shape)
    
    print('bureau and bureau balance')
    _tmp = bureau_bbalance(n_rows=n_rows)
    df = df.merge(_tmp, left_on='SK_ID_CURR', right_index=True, how='left')
    print('bureau dataframe shape:', _tmp.shape)
    del _tmp
    
    print('previous application')
    _tmp = previous_application(n_rows=n_rows)
    df = df.merge(_tmp, left_on='SK_ID_CURR', right_index=True, how='left')
    print('previous application shape:', _tmp.shape)
    del _tmp
    
    print('credit card balance')
    _tmp = credit_card_balance(n_rows=n_rows)
    df = df.merge(_tmp, left_on='SK_ID_CURR', right_index=True, how='left')
    print('credit card dataframe shape:', _tmp.shape)
    del _tmp
    
    print('installments payments')
    _tmp = installments_payments(n_rows=n_rows)
    df = df.merge(_tmp, left_on='SK_ID_CURR', right_index=True, how='left')
    print('installments dataframe shape:', _tmp.shape)
    del _tmp
    
    print('pos cash balance')
    _tmp = pos_cash_balance(n_rows=n_rows)
    df = df.merge(_tmp, left_on='SK_ID_CURR', right_index=True, how='left')
    print('pos cash dataframe shape:', _tmp.shape)
    del _tmp
    
    threshold = 0.6

    print('drop features with over {}% of missing values and ID columns'.format(round(threshold*100)))
    print(len(df.loc[:,df.isna().mean() > threshold].columns))
    #drop columns that contain more than 60% of nans
    df = df.dropna(axis=1, thresh=int(len(df) * (1 - threshold))) 
    # drop ID columns
    df = df.drop(columns=['SK_ID_CURR', 'DAYS_ID_PUBLISH'])
    print('train dataframe shape:', df.shape)
    
    # some columns contain special characters due to pd.dummies()
    # replace all non letters and numbers by '_'
    df = df.rename(columns=lambda x: re.sub('[^a-zA-Z0-9_]+', '_', x))
    print('columns are renamed')
    
    print('final merged dataframe shape:', df.shape)
    

        
#     print('drop features wiht high number of missing values and ID columns')
#     print(list(df.loc[:,df.isna().mean() > 0.60].columns))
#     #drop columns that contain more than 75% of nans
#     df = df.dropna(axis=1, thresh=len(df)*0.60, how='all') 
#     # drop ID columns
#     df = df.drop(columns=['SK_ID_CURR', 'DAYS_ID_PUBLISH'])
#     print('dataframe shape:', df.shape)
    
    #df = df.drop(columns=get_correlated_columns(df, 'TARGET', 0.85))

    #simple imputer
#     imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#     df_ = imputer.fit_transform(df)
#     df = pd.DataFrame(df_, index=df.index, columns=df.columns)

#     print('simple imputer done')
    
    # drop application test rows
    # print('drop application test rows')
    # df = df.loc[('train',),:]
    # print('final dataframe shape:', df.shape)
    
    df = reduce_memory(df)

    with open('./data/merged_dataset.pkl', 'wb') as pickle_file:
        pickle.dump(df, pickle_file)
    
    return df.loc[('train',),:], df.loc[('test',),:]
