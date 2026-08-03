import numpy as np
import pandas as pd
from IPython.display import display

from .constants import index_cols

def get_iqr_thresholds(series, threshold=1.5):
    q1 = series.quantile(0.25) # 1st quartile
    q3 = series.quantile(0.75) # 3rd quartile
    iqr = q3-q1 #InterQuartile range
    lower_limit = q1 - threshold * iqr # The minimum value 
    upper_limit = q3 + threshold * iqr # The maximum value
    return lower_limit, upper_limit


def get_std_thresholds(series, threshold=3):
    mean = series.mean()
    std = series.std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    return lower_limit, upper_limit


def get_quant_thresholds(series, threshold=0.05):
    lower_limit = series.quantile(threshold)
    upper_limit = series.quantile(1-threshold)
    return lower_limit, upper_limit


def get_thresholds(method, series):
    methods = {
        'iqr': get_iqr_thresholds(series),
        'std': get_std_thresholds(series),
        'quant': get_quant_thresholds(series),
    }
    
    assert methods.get(method), 'Unrecognized value for method, should be iqr/std/quant'
    
    return methods.get(method)


def is_outlier(series, method='iqr'):
    lower_limit, upper_limit = get_thresholds(method, series)
    
    return (series < lower_limit) | (series > upper_limit)


def non_correlated_features(dataset, target, threshold):
    corr_matrix = dataset.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_features = upper[upper[target] > threshold].index

    non_corr_features = upper.drop(columns=target).loc[:, ~upper.drop(columns=target).columns.isin(corr_features)].columns
    
    return non_corr_features


def remove_low_variance_features(df_in, threshold = .001):
    low_variance_columns = df_in.select_dtypes(include=np.number).loc[:, df.select_dtypes(include=np.number).std() < threshold].columns
    
    display(df_in[low_variance_columns].describe())

    return df_in.drop(columns=low_variance_columns)


def drop_features_with_small_variance(df_in, threshold = .001):
    numeric_columns = [col for col in df.select_dtypes(include=np.number).columns if col not in index_cols]

    features_with_small_variance = df_in[numeric_columns].columns[(df_in[numeric_columns].std(axis = 0) < threshold).values]

    corr_matrix = df_in[features_with_small_variance].join(df_in.TARGET).corr().abs()
    corr_values_target = corr_matrix['TARGET'] < threshold
    to_drop = corr_values_target[np.where(corr_values_target)[0]].index
    
    print(len(to_drop), " columns with small variance and small corr with the target are dropped" )
    print(sorted(to_drop))
    df_in = df_in.drop(to_drop, axis=1)
    
    return df_in


def small_variance(df_in, threshold = .001):
    numeric_columns = [col for col in df.select_dtypes(include=np.number).columns if col not in index_cols]

    features_with_small_variance = df_in[numeric_columns].columns[(df_in[numeric_columns].std(axis = 0) < threshold).values]

    print(features_with_small_variance)
    corr_matrix = df_in[features_with_small_variance].join(df_in.TARGET).corr().abs()
    corr_values_target = corr_matrix['TARGET'] < threshold
    to_drop = corr_values_target[np.where(corr_values_target)[0]].index
    
    print(len(to_drop), " columns with small variance and small corr with the target are dropped" )
    print(sorted(to_drop))
    df_in = df_in.drop(to_drop, axis=1)
