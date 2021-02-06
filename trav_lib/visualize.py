import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix_heat_map(df,label,qty_fields=10):
    df = pd.concat([df[label],df.drop(label,axis=1)],axis=1)
    correlation_matrix = df.corr()
    index = correlation_matrix.sort_values(label, ascending=False).index
    correlation_matrix = correlation_matrix[index].sort_values(label,ascending=False)

    fig,ax = plt.subplots()
    fig.set_size_inches((10,10))
    sns.heatmap(correlation_matrix.iloc[:qty_fields,:qty_fields],annot=True,fmt='.2f',ax=ax)
    
    # Code added due to bug in matplotlib 3.1.1
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + .5, top - .5)

    return(fig,ax)