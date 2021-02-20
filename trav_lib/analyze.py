"""Classes and functions for data exploration and analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def analyze_col(s, num_vals = 5):
    """Plot the value counts of a categorical column.
    
    Parameters
    ----------
    s: pandas series (generally using df[col])
        Series or column of dataframe to analyze
    num_vals: int
        Number of different values to plot
        
    Returns
    -------
    fig, ax: matplotlib fig and ax objects
    """
    
    num_rows = s.shape[0]
    counts = s.value_counts(dropna=False)
    
    if counts.shape[0] > num_vals:
        top = counts.head(num_vals)
        num_other_vals = counts.shape[0] - num_vals
        other_vals_name = f"Other values ({num_other_vals})"
        top[other_vals_name] = num_rows - top.sum()
    else:
        top = counts
        
    fig, ax = plt.subplots()
    fig.set_size_inches((8, top.shape[0]))
    
    top.iloc[::-1].plot.barh(ax=ax, width=0.6)
    
    x_max = top.max()
    x_sum = top.sum()
    
    x_offset = .02 * x_max
    y_offset = .15
    
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:,} ({:.1f}%)".format(int(b.x0 + b.x1), (b.x0 + b.x1) * 100 / x_sum)
        ax.annotate(val, ((b.x1) + x_offset, (b.y0) + y_offset))
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_title(s.name)
    
    return(fig, ax)
    
def bin_analyze_col(s, s_label, num_vals = 5):
    """Plot the crosstab of a categorical column with a binary label column.
    
    Plots two bar charts, the first with the value counts of each categorical
    value in s, the second with the proportion of s_label = 1 which correspond
    to each categorical value in s.
    
    Parameters
    ----------
    s: pandas series (generally using df[col])
        Series or column of dataframe to analyze
    s_label: pandas series (generally using df[col])
        Series or column of dataframe containing binary label
    num_vals: int
        Number of different values to plot
        
    Returns
    -------
    fig, (ax,ax2): matplotlib fig and ax objects
    """
    
    num_rows = s.shape[0]
    num_rows2 = s[s_label==1].shape[0]
    counts = s.value_counts(dropna=False)
    counts2 = s[s_label==1].value_counts(dropna=False)

    counts.index = counts.index.astype(str)
    counts2.index = counts2.index.astype(str)

    if counts.shape[0] > num_vals:
        top = counts.head(num_vals)
        num_other_vals = counts.shape[0] - num_vals
        other_vals_name = f"Other values ({num_other_vals})"

        top2 = counts2.loc[top.index]

        top[other_vals_name] = num_rows - top.sum()
        top2[other_vals_name] = num_rows2 - top2.sum()

    else:
        top = counts
        top2 = counts2.loc[top.index]

    top2 = top2 / top

    fig, (ax, ax2) = plt.subplots(1,2)
    fig.set_size_inches((16, top.shape[0]))
    fig.suptitle(s.name)

    top.iloc[::-1].plot.barh(ax=ax, width=0.6)

    x_max = top.max()
    x_sum = top.sum()

    x_offset = .02 * x_max
    y_offset = .15

    for p in ax.patches:
        b = p.get_bbox()
        val = "{:,}\n({:.1f}%)".format(int(b.x0 + b.x1), (b.x0 + b.x1) * 100 / x_sum)
        ax.annotate(val, ((b.x1) + x_offset, (b.y0) + y_offset))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title('Total Category Frequency')

    top2.iloc[::-1].plot.barh(ax=ax2, width=0.6, color = '#dd8552')

    ax2.set_xlim(0,1)
    
    x_max = top2.max()
    x_sum = top2.sum()

    x_offset = .02 * x_max
    y_offset = .15

    for p in ax2.patches:
        b = p.get_bbox()
        val = "{:.1f}%".format((b.x0 + b.x1)*100)
        ax2.annotate(val, ((b.x1) + x_offset, (b.y0) + y_offset))

    ax2.set_title('Label = 1 Proportion by Category')

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    return(fig, (ax, ax2))