import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import loguniform

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from .constants import RND_SEED

def get_xy_reduced(X, y, ratio=0.2):
    print('*'*20)
    print('SAMPLED MODE\nONLY {}% OF THE DATA IS USED\nTHE REDUCED DATA HAVE THE SAME TARGET DISTRIBUTION'.format(ratio*100))
    print('*'*20)

    X_reduced, _, y_reduced, _ = train_test_split(
        X,
        y,
        train_size=ratio,
        stratify=y,
        random_state=RND_SEED,
    )

    return X_reduced, y_reduced


def correlatedFeatures(dataset, target, threshold):
    corr_matrix = dataset.join(target).corr().abs()

    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Sort features to select the column than is less correlated to the target
    features_sorted = upper[target.name].sort_values(na_position='first').drop(target.name).keys() # drop column TARGET
    upper = upper.drop(target.name)[features_sorted] # drop row TARGET
    
    correlated_columns = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return correlated_columns


def resampling(X, y, method):
    if (method == 'SMOTE'):
        return SMOTE(random_state=RND_SEED).fit_resample(X, y)
    else:
        # Put X and y training data back together again
        Xy = pd.concat([X, y], axis=1)
        # Split into repaided and defaulted
        Xy_repaided = Xy[Xy[y.name]==0]
        Xy_defaulted = Xy[Xy[y.name]==1]
        
        oversampling = method == 'oversampling'

        if oversampling:
            to_resample = Xy_defaulted
            no_change = Xy_repaided
        else: 
            to_resample = Xy_repaided
            no_change = Xy_defaulted

        Xy_sampled = resample(to_resample, 
                                    replace=oversampling, 
                                    n_samples=no_change.shape[0],
                                    random_state=RND_SEED)

        # Combine the two classes
        combined = pd.concat([Xy_sampled, no_change])

        # Re-split the training data
        y = combined[y.name]
        X = combined.drop(y.name, axis=1)

        return X, y


def kFoldCV(X, y, model, silent=False):
    """Evaluate model with k-fold cross validation"""
        
    # Create folds
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RND_SEED)
    
    # Perform kfold cross validation
    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    # Show results
    if not silent:
        print(type(model).__name__)
        print("kFoldCV:")
        print("    Fold R2 scores:", results)
        print("    Mean R2 score:", results.mean())
        print("    Std R2 score:", results.std())
    
    # Build a model using all the data and return it
    model.fit(X, y)

    return model


def rfecv_selection(X, y, min_features_to_select, step, debug=False):
    if debug:
        X, y = get_xy_reduced(X, y, ratio=0.2)
        
    estimator = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), # Step1 - impute missing values
        ('scaler', MinMaxScaler(feature_range=(0,1))), # Step2 - normalize data
        ("smote", SMOTE(random_state=RND_SEED)),
        ("model", RandomForestClassifier(n_jobs=-1, random_state=RND_SEED)
        ),
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RND_SEED)
    selector = RFECV(estimator, step=step, cv=cv, scoring='roc_auc', min_features_to_select=min_features_to_select,verbose=3, n_jobs=-1,importance_getter=("named_steps.model.feature_importances_"))
    selector.fit(X, y)

    print("Optimal number of features : %d" % selector.n_features_)
    print(selector.support_)
    
    removed_features  = list(X.columns[~selector.support_])
    #print("Feature selection", selector.support_)
    #print("Feature ranking", selector.ranking_)
    #print("Selected features:", selected_features)
    #print("Removed features:", list(X.columns[~selector.support_]))
    
    return removed_features , selector


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y=None):
        """ Get either categorical or numerical columns on fit.
        Store as attribute for future reference"""
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cat_cols = X.columns[X.apply(lambda x: all(x.dropna()%1==0))].tolist()

        if self.dtype == 'numerical':
            #self.cols = X.select_dtypes(exclude='O').columns.tolist()
            self.cols = [x for x in X.columns.tolist() if x not in cat_cols]
        elif self.dtype == 'categorical':
            self.cols = cat_cols
            
        self.col_idx = [X.columns.get_loc(col) for col in self.cols]
        return self

    def transform(self, X):
        """ Subset columns of chosen data type and return np.array"""
        X = X.values if isinstance(X, pd.DataFrame) else X
        return X[:, self.col_idx]

def evaluate_models(X, y, methods, pipeline, debug=False):
    if debug:
        X, y = get_xy_reduced(X, y, ratio=0.2)
        
    #evaluation - baselines
    num_folds = 10
    shuffle=True
    scoring = 'roc_auc'

    results = []
    names = []
    for classifier in methods:
        pipeline.set_params(clf = classifier)

        kfold = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=RND_SEED)
        scores = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring,n_jobs=-1)
        results.append(scores['test_score'])
        names.append(type(pipeline['clf']).__name__)
        print(str(classifier))
        for key, values in scores.items():
            print(key, 'mean:', round(values.mean(),3), 'std:', round(values.std(),4))
        
    return results, names


def compare_models(results, names):
    # compare algorithms
    fig = plt.figure()
    fig.suptitle('Comparison of Models')
    ax = fig.add_subplot(111)
    plt.boxplot(results, vert=False)
    ax.set_yticklabels(names)
    ax.set_xlabel('ROC AUC score')
    plt.show();


def model_performance(model, X_test, y_test):
    # Check the model performance with the test data
    predictions = model.predict(X_test)

    # Plot the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nROC AUC:", roc_auc_score(y_test, predictions))
    
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=['repayed', 'defaulted']))
    # ROC / AUC
    plotRocAuc(model, X_test, y_test)


def plotRocAuc(model, X, y):
    
    probabilities = model.predict_proba(X)
    probabilities = probabilities[:, 1]  # keep probabilities for first class only
        
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y, probabilities)    
    
    # Plot the "dumb model" line
    plt.plot([0, 1], [0, 1], linestyle='--')
    
    # Plot the model line
    plt.plot(fpr, tpr, marker='.')
    plt.text(0.75, 0.25, "AUC: " + str(round(roc_auc_score(y, probabilities),2)))
    
    # show the plot
    plt.show()


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


def get_hyperparameters_random(X, y, debug=False):
    if debug:
        X, y = get_xy_reduced(X, y, ratio=0.2)
        
    pipe = Pipeline([
        ('scaler', MinMaxScaler()), # Step2 - normalize data
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=RND_SEED)) # Step4 - classification
    ])

    
    
    param_distributions = {
        #'clf__l2_regularization': loguniform(1e-6, 1e3),
        'clf__l2_regularization': loguniform(250, 1000),
        'clf__learning_rate': loguniform(.001, .025),
        'clf__max_leaf_nodes': loguniform_int(2, 100),
        'clf__min_samples_leaf': loguniform_int(1, 100),
        'clf__max_depth' :  loguniform_int(10, 100),
        'clf__max_iter' :   loguniform_int(200, 1000),


        #'clf__max_bins': loguniform_int(2, 255),
    }

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RND_SEED)

    
    random_search = RandomizedSearchCV(
        pipe, param_distributions=param_distributions, n_iter=250,
        cv=kfold, verbose=3, n_jobs=3, scoring='roc_auc',
        random_state=RND_SEED)

    random_search.fit(X, y)

    print(random_search.best_score_)
    print(random_search.best_estimator_)
    print(random_search.best_params_)
    
    return random_search


def get_hyperparameters(X, y, debug=False):
    if debug:
        X, y = get_xy_reduced(X, y, ratio=0.2)

    pipe = Pipeline([
        ('scaler', MinMaxScaler()), # Step2 - normalize data
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=RND_SEED)) # Step4 - classification
    ])

    param_grid = { 
       'clf__max_iter': [1000],
       #  'clf__max_iter': [1000],
        #'clf__learning_rate': [.01, .1, .2],
        'clf__learning_rate': [.025, 0.015, .01],
        'clf__max_depth' : [50, 75, 100],
       # 'clf__max_leaf_nodes' : [10, 30],
        'clf__min_samples_leaf' : [50, 30,2],
        'clf__l2_regularization': [250,500,750,1000],
    }

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RND_SEED)

    # Search for best hyperparameters
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='roc_auc', verbose=1, n_jobs=3)
    grid_search.fit(X, y)

    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    
    return grid_search
