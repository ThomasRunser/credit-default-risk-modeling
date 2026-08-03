import pickle

from .constants import (
    BUREAU_AGG,
    CREDIT_CARD_AGG,
    PREVIOUS_APPLICATION_AGG,
    POS_CASH_AGG,
    INSTALLMENTS_AGG,
    index_cols,
)
from .feature_engineering import one_hot_encoder, df_aggregations

def bureau_bbalance(n_rows):
    with open('./data/bureau.pkl', 'rb') as pickle_file:
        bureau = pickle.load(pickle_file)
    with open('./data/bureau_balance.pkl', 'rb') as pickle_file:
        bbalance = pickle.load(pickle_file)
        
    ### Debug Mode ###
    bureau = bureau.iloc[:n_rows,:]
    bbalance = bbalance.iloc[:n_rows,:]
        
    # Months with late payments (days past due)
    bbalance['PAST_DUE'] = bbalance['STATUS'].transform(lambda x: 1 if x in ['1', '2', '3', '4', '5'] else 0)
    bbalance['ON_TIME'] = bbalance['STATUS'].transform(lambda x: 1 if x == '0' else 0)
    
    bbalance = bbalance.drop(columns=['STATUS'])

    bbalance, bbalance_cat = one_hot_encoder(bbalance, nan_as_category=False)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=False)
    
    bbalance_cat += ['PAST_DUE', 'ON_TIME']

    bbalance_agg = df_aggregations(bbalance, bbalance_cat, 'SK_ID_BUREAU', 'bbalance', index_cols)

    bbalance_features_aggregations = {}
    for f in bbalance_agg.columns: bbalance_features_aggregations[f] = [f.split('_')[-1]]
    
    bbalance_features_aggregations.update(BUREAU_AGG)

    bureau = bureau.join(bbalance_agg, how='left', on='SK_ID_BUREAU')
    bureau_agg = df_aggregations(bureau, bureau_cat, 'SK_ID_CURR', 'bureau', index_cols, bbalance_features_aggregations)

    return bureau_agg


def previous_application(n_rows):
    with open('./data/previous_application.pkl', 'rb') as pickle_file:
        prev = pickle.load(pickle_file)
        
    ### Debug Mode ###
    prev = prev.iloc[:n_rows,:]
    
    ### OHE ###
    prev, prev_cat = one_hot_encoder(prev, nan_as_category=False)
    
    ### Enegiring Features ### 
    prev['DAYS_LAST_DUE_DIFFERENCE'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    
    prev_agg = df_aggregations(prev, prev_cat, 'SK_ID_CURR', 'prev', index_cols, PREVIOUS_APPLICATION_AGG)
    
    return prev_agg


def credit_card_balance(n_rows):    
    with open('./data/credit_card_balance.pkl', 'rb') as pickle_file:
        cc = pickle.load(pickle_file)
        
    ### Debug Mode ###
    cc = cc.iloc[:n_rows,:] 
    
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    
    cc, cc_cat = one_hot_encoder(cc, nan_as_category=False)
    
    cc_cat += ['LATE_PAYMENT']
    
    cc_agg = df_aggregations(cc, cc_cat, 'SK_ID_CURR', 'cc', index_cols, CREDIT_CARD_AGG)
    
    return cc_agg


def installments_payments(n_rows):
    with open('./data/installments_payments.pkl', 'rb') as pickle_file:
        pay = pickle.load(pickle_file)
        
    ### Debug Mode ###
    pay = pay.iloc[:n_rows,:] 
    
    pay['LATE_PAYMENT'] = pay['DAYS_ENTRY_PAYMENT'] < pay['DAYS_INSTALMENT']
    pay['LATE_PAYMENT'] = pay['LATE_PAYMENT'] .transform(lambda x: 1 if x==True else 0)
    
    pay, pay_cat = one_hot_encoder(pay, nan_as_category=False)
    
    pay_cat += ['LATE_PAYMENT']
    
    pay_agg = df_aggregations(pay, pay_cat, 'SK_ID_CURR', 'pay', index_cols, INSTALLMENTS_AGG)
    
    return pay_agg


def pos_cash_balance(n_rows):
    with open('./data/POS_CASH_balance.pkl', 'rb') as pickle_file:
        cash = pickle.load(pickle_file)
        
    ### Debug Mode ###
    cash = cash.iloc[:n_rows,:] 
    
    cash['LATE_PAYMENT'] = cash['SK_DPD'].transform(lambda x: 1 if x > 0 else 0)
    
    cash, cash_cat = one_hot_encoder(cash, nan_as_category=False)
    
    cash_cat += ['LATE_PAYMENT']
    
    cash_agg = df_aggregations(cash, cash_cat, 'SK_ID_CURR', 'cash', index_cols, POS_CASH_AGG)
    
    return cash_agg
