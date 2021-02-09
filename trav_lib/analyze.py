"""Classes and functions for data exploration and analysis."""

import pandas as pd
import numpy as np

def top_value_counts(df, n=5, only_categories = True, cols_to_include = None, cols_to_exclude = None):
    """ Function to generate summary information for string or categorical
    data in dataframes"""

    if cols_to_include:
        df = df[cols_to_include]
    if cols_to_exclude:
        df = df[df.columns[~df.columns.isin(cols_to_exclude)]]
        
    if 'float' in list(df.dtypes):
        print("Error, column(s) with float dtype included")
        print('The following columns will be excluded',list(df.select_dtypes(include='float64').columns))
    
    df = df.select_dtypes(exclude='float64')
    
    if only_categories:
        df = df.select_dtypes(include=['O','category'])
        
    cols = df.columns
    df_value_counts = pd.DataFrame()
    i_name = -1
    for col in cols:
        i_name += 1
        counts = df[col].value_counts(dropna=False)[:n]
        top_n_names = list(counts.index)
        top_n = list(counts)
        if len(top_n) < n+1:
            for i in range(n-len(top_n)):
                top_n.append('-')
                top_n_names.append('-')
        top_n_names.insert(0,'n_unique')
        top_n.insert(0,df[col].nunique())
        df_value_counts[col] = top_n_names
        df_value_counts[i_name] = top_n
    new_index = pd.MultiIndex.from_product([df_value_counts.columns[range(0,len(df_value_counts.columns),2)],('Cat','Freq')],names = ['field','info'])
    df_value_counts.columns = new_index
    return(df_value_counts)