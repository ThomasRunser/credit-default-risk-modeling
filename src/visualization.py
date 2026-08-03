import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas.api.types import is_numeric_dtype

def axis_percent(ax = None, axis=1, digits=0):
    if ax is None:
        ax = plt.gca()
    def _to_percent(x, position):
        return '{:.{}f}%'.format(x*100, digits)
    formatter = FuncFormatter(_to_percent)

    if axis == 0:
        ax.xaxis.set_major_formatter(formatter)        
    elif axis == 1:
        ax.yaxis.set_major_formatter(formatter)
    elif axis == 2:
        ax.zaxis.set_major_formatter(formatter)


def display_discrete(df, x_vars, y_var, kind='kde'):  
    # convert the dataframe from wide to 
    dfm = df[x_vars+[y_var]].loc[:,df[x_vars+[y_var]].std()>0].melt(id_vars=y_var, var_name='Distribution')
    dfm.sort_values(['Distribution'], inplace=True)
    
    if kind == 'hist':
        g = sns.displot(kind='hist', data=dfm, col='Distribution', col_wrap=8, x='value', hue='TARGET', fill=True, palette='tab10', facet_kws={'sharey': True, 'sharex': True}, 
                        discrete=True, height=2.5, aspect=1)
        plot_kind = 'Histplot'
    else:
        g = sns.displot(kind='kde', data=dfm, col='Distribution', col_wrap=3, x='value', hue='TARGET', fill=True, palette='tab10', warn_singular=False, facet_kws={'sharey': False, 'sharex': False}, 
                        height=2.5, aspect=2)
        plot_kind = 'Kdeplot'
                                                                                 
    for ax in g.axes:
        ax.margins(x=0)
        ax.margins(y=0)

    g.set_titles("{col_name}")
    g.fig.suptitle('{} of the discrete and TARGET features'.format(plot_kind), fontsize=16, y=1.01)
    g.tight_layout()


def pie_chart(df):
    #plt.rcParams.update({'font.size': 22})
    ax, fig = plt.subplots(facecolor='white', figsize=(8, 8))
    ax.text(0.5,0.5,'Loan', ha="center", va="center", fontsize=32) 

    palette_color = sns.color_palette('tab10')

    values = df['TARGET'].value_counts()

    # plotting data on chart
    fig = plt.pie(values, colors=palette_color, radius=2, autopct='%0.1f%%', startangle = 60,  pctdistance=0.75)
    my_circle = plt.Circle((0,0), 1, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    ax.legend(['Repayed', 'Defaulted'], loc='lower right')

    plt.title('Loan Repayed or not',y=1.4, fontsize=32) 

    plt.show()


def display_categorical(df, x_vars, y_var, height=5, aspect=1, orient='h'):
    if orient=='h':
        n = 4 if len(x_vars) > 4 else len(x_vars)
        rotation = 0
        xy_percent = 0
        sharey, sharex = False, True
        x, y = None, 'value'
        x_label, y_label = 'Percent', ''
    else:
        n = 2 if len(x_vars) > 2 else len(x_vars)
        rotation = 90
        xy_percent = 1
        sharey, sharex = True, False
        x, y = 'value', None
        x_label, y_label = '', 'Percent'

        
    # convert the dataframe from wide to 
    dfm = df[x_vars+[y_var]].melt(id_vars=y_var, var_name='Distribution')
    dfm.sort_values(['Distribution','value'], inplace=True)
    
    # plot
    g = sns.displot(kind='hist', data=dfm, col='Distribution', col_wrap=n, 
                    stat='percent', x=x, y=y, multiple='fill', fill=True,
                    hue=y_var, palette='tab10', 
                    facet_kws={'sharey': sharey, 'sharex': sharex}, height=height, aspect=aspect
                   )

    for ax in g.axes:
        axis_percent(ax, axis=xy_percent)

        # add annotations
        for c in ax.containers:
            labels = [f"{int(v * 100)}%" if (v > 0.15) else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center', rotation=rotation)
        ax.tick_params(axis='x',rotation=rotation)
        ax.margins(x=0)
        ax.margins(y=0)

    g.set_axis_labels(x_label,y_label)    
    g.set_titles("{col_name}")
    g.fig.suptitle('Bar Plot of the categorical and TARGET features', fontsize=16, y=1.01)
    g.tight_layout()


def display_correlated_data(df):
    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with correct aspect ratio
    sns.heatmap(corr,cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlated data')
    plt.show()


def get_correlated_columns(df, target, threshold):

    corr_matrix = df.corr(numeric_only=True).abs()

    features_sorted = corr_matrix.loc[target,:].sort_values(na_position='first', ascending=True).drop(target).keys() # drop column TARGET
    corr_matrix = corr_matrix.drop(target, axis=0)[features_sorted] # drop row TARGET

    #print(df[['TARGET','AGE_GROUP','DAYS_BIRTH']].corr().abs())
    
    correlated_columns = []
    for c in features_sorted.to_list():
        if any(corr_matrix.loc[~corr_matrix.index.isin([c]),c] > 0.85):
            corr_matrix.drop(c, inplace=True, axis=0)
            correlated_columns.append(c)
    
    return correlated_columns


def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''

    # Get a list of class names
    numclasses = len(model.classes_)
    if len(model.classes_)==2:
        classes =  [model.classes_[1]] # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_

    # Create a plot for each class
    for i,c in enumerate(classes):
        # Plot the coefficients as bars
        fig = plt.figure(figsize=(8,len(column_names)/4))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16)
        rects = plt.barh(column_names, model.coef_[i],color="lightblue")
        
        # Annotate the bars with the coefficient values
        for rect in rects:
            width = round(rect.get_width(),4)
            plt.gca().annotate('  {}  '.format(width),
                        xy=(0, rect.get_y()),
                        xytext=(0,2),  
                        textcoords="offset points",  
                        ha='left' if width<0 else 'right', va='bottom')    
        plt.tight_layout()
        plt.show()


def decisionTreeSummary(model, column_names):
    '''Show a summary of the trained decision tree model'''

    # Plot the feature importances as bars
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Decision tree feature importance', fontsize=16)
    rects = plt.barh(column_names, model.feature_importances_,color="khaki")

    # Annotate the bars with the feature importance values
    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(width, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left', va='bottom')    

    plt.show()


def display_distribution_old(df, columns):
    row = len(columns) + 1
    plt.figure(figsize = (16, row * 4))
    for i, column in enumerate(columns, 1):
        if is_numeric_dtype(df[column]):
            plt.subplot(row,2,i*2+1)
            sns.kdeplot(data=df, x=column, palette="Set2")
            plt.subplot(row,2,i*2+2)
            sns.boxplot(data=df, x=column, orient='h', palette="Set2")
        else:
            plt.subplot(row,2,i*2+1)
            sns.histplot(data=df, y=column, palette="Set2")
    plt.show()


def display_rfecv(min_features_to_select, step, n_features, cv_results):
    
    n_scores = len(cv_results["mean_test_score"])
    range_ = list(range(n_features, min_features_to_select, -step))

    if range_[-1] != min_features_to_select:
        range_.append(min_features_to_select)

    range_.reverse()

    plt.figure(figsize=(10,10))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test roc auc")
    plt.errorbar(
        range_,
        cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()
