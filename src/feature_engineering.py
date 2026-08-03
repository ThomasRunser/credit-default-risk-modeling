import math
import pickle
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .constants import (
    DROP_APPLICATION_FEATURES,
    DROP_APPLICATION_FEATURES2,
    index_cols,
)
from .outliers import is_outlier

def poly_features(df_in):
    #Make a new dataframe for polynomial features
    poly_features_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    poly_features = df_in[poly_features_cols].copy()
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    poly_features_ = imputer.fit_transform(poly_features)
    poly_features = pd.DataFrame(poly_features_, index=poly_features.index, columns=poly_features.columns)

    #print('**'*50)

    from sklearn.preprocessing import PolynomialFeatures

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree = 3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features_ = poly_transformer.transform(poly_features)
    #print('Polynomial Features shape: ', poly_features_.shape)
    #print('**'*20)
    # Create a dataframe of the features 
    poly_features = pd.DataFrame(poly_features_, index=poly_features.index, columns=poly_transformer.get_feature_names_out())

    # Find the correlations with the target
    poly_corrs = poly_features.join(df_in['TARGET']).corr()['TARGET'].sort_values()

    # Display most negative and most positive
    #print(poly_corrs.head(10))
    #print('**'*50)
    #print(poly_corrs.tail(10))

    poly_features['SK_ID_CURR'] = df_in['SK_ID_CURR']
    poly_features.set_index('SK_ID_CURR', inplace=True)
    poly_features.drop(columns=([*poly_features_cols, '1']), inplace=True)
    
    return poly_features


def one_hot_encoder(df, nan_as_category=True):
    original_columns = df.columns
    df = pd.get_dummies(df, dummy_na=nan_as_category)
    new_columns = [col for col in df.columns if col not in original_columns]
    return df, new_columns


def features_aggregations(df, features_cat, ignore_cols):
    features_num = [col for col in df.columns if col not in (*features_cat, *ignore_cols)]

    features_aggregations = {}

    for f in features_cat: features_aggregations[f] = ['max']
    for f in features_num: features_aggregations[f] = ['mean', 'max', 'min', 'sum']
        
    return features_aggregations


def df_aggregations(df, df_cat, groupby_col, df_name, ignore_cols, aggs = None):
    aggregations = features_aggregations(df, df_cat, ignore_cols)
    if aggs != None:
        aggregations.update(aggs) 
 
    df_agg = df.groupby(groupby_col).agg(aggregations)
    df_agg.columns = pd.Index(['{}_{}_{}'.format(df_name, index[0], index[1]) for index in df_agg.columns])
    
    return df_agg



def create_application_visualization_dataset():
    with open('./data/application_train.pkl', 'rb') as pickle_file:
        train = pickle.load(pickle_file)

    with open('./data/application_test.pkl', 'rb') as pickle_file:
        test = pickle.load(pickle_file)

    test = test.copy()  # rebuilds the DataFrame in contiguous memory
    test["TARGET"] = np.nan
    # Merge application_train and application_test
    df = pd.concat([test, train], keys=['test', 'train']).copy()

    # Remove accidental spaces at the beginning or end of column names
    df.columns = df.columns.str.strip()

    age_groups = [24, 32, 39, 53, 65]
    df['AGE_GROUP'] = df['DAYS_BIRTH'].apply(lambda x: get_age_group(x, age_groups))

    with open('./data/application_visualization.pkl', 'wb') as pickle_file:
        pickle.dump(df, pickle_file)

    return df

def get_age_group(days_birth, age_groups):
    age_years = np.floor(-days_birth / 365)
    count = 0
    for upper_limit in age_groups:
        if age_years < upper_limit: 
            return count
        count+=1
    return count


def get_age_groups(df):
    age_groups, score = [], 0
    _age_groups = []

    age_min = math.trunc(-df['DAYS_BIRTH'].max() / 365)
    age_max = math.trunc(-df['DAYS_BIRTH'].min() / 365)

    while True:
        age_pairs = pairwise([age_min]+_age_groups+[age_max])

        for lower_limit, upper_limit in age_pairs:
            for i in range(lower_limit+1, upper_limit-1):
                new_age_groups = sorted(_age_groups + [i])
                _series = df['DAYS_BIRTH'].apply(lambda x: get_age_group(x, new_age_groups))
                _score = _series.corr(df['TARGET'])

                if score > _score:
                    score = _score
                    age_groups = new_age_groups

        if age_groups == _age_groups:
            break
        print(age_groups, score)
        _age_groups = age_groups

        
    return age_groups
