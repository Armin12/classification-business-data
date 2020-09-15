import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def two_labels_data_scatter_plotter(concat_df, colx, coly, col_label, label1, label2,
                                    xmin, xmax, ymin, ymax, filename='x-y-two-label.png'):
    """Scatter plot of two columns of company data for two specific labels. 
    The data related to one label is blue and for the other one is red.
    
    Parameters
    ----------
    concat_df : pandas.DataFrame
        Dataset
    colx : str
        Column x
    coly : str
        Column y
    col_label : str
        Label column
    label1, label2 : str
        Labels to compare their data
    xmin, xmax, ymin, ymax : float
        Minimum and maximum values on the x and y axes
    filename : str
        Name of the image file to be saved
    
    Returns
    ----------
    figure.Figure
    """

    x = concat_df[colx]
    y = concat_df[coly]
    l = concat_df[col_label]

    x1 = x[l == label1]
    x2 = x[l == label2]

    y1 = y[l == label1]
    y2 = y[l == label2]

    r1, p1 = pearsonr(x1, y1)
    r2, p2 = pearsonr(x2, y2)

    fig = plt.figure(figsize=(12, 10))
    plt.scatter(x1, y1, marker="^", color='b')
    plt.scatter(x2, y2, marker="^", color='r')
    plt.ylabel(coly, fontweight='bold', labelpad=20)
    plt.xlabel(colx, fontweight='bold', labelpad=20)
    plt.axis('tight')
    plt.tight_layout()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend((label1 + ' Corr={0}'.format(np.round_(r1, 3)), label2 + ' Corr={0}'.format(np.round_(r2, 3))),
               loc='upper right', fontsize=30)
    plt.show()
    fig.savefig(filename, dpi=350)


def label_averaged_data_scatter_plotter(concat_df, colx, coly, col_label,
                                        xmin, xmax, ymin, ymax, filename='x-y-label-averaged.png'):
    """Scatter plot of label-averaged data for two columns.
    
    Parameters
    ----------
    concat_df : pandas.DataFrame
        Dataset
    colx : str
        Column x
    coly : str
        Column y
    col_label : str
        Label column
    xmin, xmax, ymin, ymax : float
        Minimum and maximum values on the x and y axes
    filename : str
        Name of the image file to be saved
    
    Returns
    ----------
    figure.Figure
    """

    label_mean = concat_df[[colx, coly, col_label]].groupby(col_label).mean().reset_index()

    x = label_mean[colx]
    y = label_mean[coly]

    r, p = pearsonr(x, y)

    fig = plt.figure(figsize=(12, 10))
    plt.scatter(x, y, marker="^", color='b')
    plt.ylabel(coly, fontweight='bold', labelpad=20)
    plt.xlabel(colx, fontweight='bold', labelpad=20)
    plt.axis('tight')
    plt.tight_layout()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.text(75, 9e6, 'Corr={0}'.format(np.round_(r, 3)), fontsize=26)
    plt.show()
    fig.savefig(filename, dpi=350)


def df_time_series_plotter(df, filename='time-y.png'):
    """Time series plots of an arbitrary dataframe with time presented across columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    filename : str
        Filename
    
    Returns
    ----------
    figure.Figure
    """

    fig = plt.figure(figsize=(12, 10))
    df.T.plot(kind='line')
    plt.xlabel('Time', fontweight='bold', labelpad=25)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()
    fig.savefig(filename, dpi=350)
