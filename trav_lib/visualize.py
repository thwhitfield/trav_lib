"""Classes and functions used for data visualization"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_matrix_heat_map(df, label, qty_fields=10):
    df = pd.concat([df[label], df.drop(label, axis=1)], axis=1)
    correlation_matrix = df.corr()
    index = correlation_matrix.sort_values(label, ascending=False).index
    correlation_matrix = correlation_matrix[index].sort_values(label, ascending=False)

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    sns.heatmap(
        correlation_matrix.iloc[:qty_fields, :qty_fields], annot=True, fmt=".2f", ax=ax
    )

    # Code added due to bug in matplotlib 3.1.1
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    return (fig, ax)


def plot_log_hist(s, bin_factor=1, min_exp=None):
    """Plot 2 histograms with log x scales, one for positive values & one for negative values.

    Bin_factor is used to scale how many bins to use (1 is default and corresponds to
    one bin per order of magnitude. Higher than 1 will skew the bins away from even powers of
    10).

    Parameters
    ----------
    s: pandas series (generally using df[col])
        Series or column of dataframe to analyze
    bin_factor: int
        Default 1, used to scale how many bins to use
    min_exp: int
        The minimum exponent to use in creating bins & plotting.
        This can be set manually for cases where you want a specific
        minimum value to be shown.

    Returns
    -------
    fig, (ax1,ax2): matplotlib fig and ax objects
    """

    # Split series into positive & negative components
    s_pos = s[s >= 0]
    s_neg = s[s < 0].abs()

    # Not the best way to deal with this, but this was the easiest solution for now.
    # TODO Fix this code to deal with no negative values or no positive values more appropriately
    if s_neg.shape[0] == 0:
        s_neg.loc[0] = 1
    if s_pos.shape[0] == 0:
        s_pos.loc[0] = 1

    # Calculate appropriate min_exp if none provied
    if min_exp == None:
        threshold = s_pos.shape[0] - (s_pos == 0).sum()

        for i in range(10):
            n_betw = s_pos[s_pos != 0].between(0, 10**-i).sum()
            if not (n_betw / threshold) > 0.1:
                min_exp = -i
                break

    # Clip values to the 10**min_exp so that they are included in the histograms (if
    # this isn't done then values which are 0 will be excluded from the histogram)
    s_pos = s_pos.clip(lower=10**min_exp)
    s_neg = s_neg.clip(lower=10**min_exp)

    # Calculate the lowest integer which encompases all the positive and negative values
    pos_max = int(np.ceil(np.log10(max(s_pos))))
    neg_max = int(np.ceil(np.log10(max(s_neg))))

    # Use that for both negative & positive values
    plot_max = max(pos_max, neg_max)

    # Create the bins (bin spacing is logarithmic)
    bins = np.logspace(min_exp, plot_max, (plot_max + 1) * bin_factor)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_size_inches((10, 5))

    s_neg.hist(bins=bins, ax=ax1)
    ax1.set_xscale("log")
    ax1.set_title("Distribution of Negative Values")
    ax1.set_xlabel("Negative values")

    s_pos.hist(bins=bins, ax=ax2)
    ax2.set_xscale("log")
    ax2.set_title("Distribution of Positive Values")
    ax2.set_xlabel("Positive Values")

    # Invert axis so that values are increasingly negative from right to left.
    # Decrease the spacing between the two subplots
    ax1.invert_xaxis()
    plt.subplots_adjust(wspace=0.02)

    return (fig, (ax1, ax2))
