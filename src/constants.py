RND_SEED = 7

index_cols = [
    "TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV",
    "index", "DAYS_ID_PUBLISH",
]

DROP_APPLICATION_FEATURES = [
    'FLAG_MOBIL',
    'FLAG_CONT_MOBILE',
    'FLAG_EMAIL',
    'REG_REGION_NOT_LIVE_REGION',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12',
    'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20',
    'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT']
DROP_APPLICATION_FEATURES2 = ['EXT_SOURCE_1_2',
 'EXT_SOURCE_1_3']
BUREAU_AGG = {
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'CREDIT_DAY_OVERDUE': [],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'DAYS_ENDDATE_FACT':[],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'], 
    'CNT_CREDIT_PROLONG':[],
    'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_LIMIT':[],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
    'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
    'AMT_ANNUITY': ['mean'],
}
CREDIT_CARD_AGG = {
    'MONTHS_BALANCE': ['min'], # negative value
    'AMT_BALANCE': ['max'],
    'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
    'AMT_DRAWINGS_ATM_CURRENT': ['sum'],
    'AMT_DRAWINGS_CURRENT': ['sum'],
    'AMT_DRAWINGS_OTHER_CURRENT': ['sum'],
    'AMT_DRAWINGS_POS_CURRENT': ['sum'],
    'AMT_INST_MIN_REGULARITY': ['sum', 'mean'],
}
PREVIOUS_APPLICATION_AGG = {
    'DAYS_FIRST_DRAWING': [],
    'DAYS_FIRST_DRAWING': [],
    'DAYS_LAST_DUE_1ST_VERSION': [],
    'DAYS_LAST_DUE': [],
    'DAYS_FIRST_DUE': [],
    'NFLAG_LAST_APPL_IN_DAY': [],
    'NFLAG_INSURED_ON_APPROVAL': ['max'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
}
POS_CASH_AGG = {
    'SK_DPD_DEF': ['max', 'mean', 'sum'], #mean OK
    'SK_DPD': ['max', 'mean', 'sum'],  #mean OK
}
INSTALLMENTS_AGG = {
    'NUM_INSTALMENT_VERSION': [],
    'NUM_INSTALMENT_NUMBER': [],
}
