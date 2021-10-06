import pandas as pd
import numpy as np
import re

def reduce_memory(df, cat_cols = None, verbose = True):
    """Reduce dtype size of dataframe columns to smallest available & convert specified columns
    to categorical.
    
    Parameters
    ----------
    df: pandas dataframe
        dataframe which will be reduced in size
    cat_cols: list (default: None)
        list of column names to convert to categorical
    verbose: bool (default: True)
    
    Returns
    -------
    df: pandas dataframe
        dataframe after it has been reduced in size
    """
    
    if verbose:
        memory_before = df.memory_usage(deep=True).sum() / 1e6
        print(f'before: {memory_before:.1f} MB')

    if cat_cols is None:
        cat_cols = []

    for col in cat_cols:
        df[col] = df[col].astype(str).astype('category')

    for col in df.columns.drop(cat_cols):
        col_type = str(df[col].dtype)
        if 'int' in col_type:
            df[col] = pd.to_numeric(df[col], downcast = 'integer')

        elif 'float' in col_type:
            df[col] = pd.to_numeric(df[col], downcast = 'float')
        
    if verbose:
        memory_after = df.memory_usage(deep=True).sum() / 1e6
        print(f'after: {memory_after:.1f} MB')
        print(f'decreased by: {100*(memory_before - memory_after) / memory_before:.1f}%')
    
    return(df)

def snake_case(word):
    """Convert word to snake_case
    
    Parameters
    ----------
    word: string
    
    Returns
    -------
    word: string
    """
    
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_").replace(' ','_')
    word = word.lower()
    
    return(word)
                  