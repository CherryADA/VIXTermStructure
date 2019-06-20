# IMPORT MODULES

import pandas as pd

import numpy as np

import math

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# DEFINE USEFUL FUNCTIONS

def clean_df(m):
    """

    Take the master dataframe with all of the information, and extract a

    sub-dataframe which consists of columns m and m+1. Then drop NaN values.

    Label rows with the dates (column m), and convert string values to

    numeric.

    """

    a = pd.to_numeric(df.iloc[:, [m, m + 1]].set_index('Date').dropna().iloc[:, 0])

    return a


def df_merge(lodf):
    """

    Take a list of data frames (lodf), and merge all of them into one

    single dataframe. This new dataframe will contain rows that are labelled

    by dates that appear in every dataframe in the lodf.

    The column labels of this new dataframe will be names of the original

    dataframes in lodf.

    Note: This function also works for a list of Series.

    This code is meant to be used on a list of Series. I haven't tested it on

    a list of dataframes which have more than one column.

    """

    n = len(lodf)

    i = 2

    labels = [lodf[0].name, lodf[1].name]

    newdf = pd.merge(lodf[0], lodf[1], left_index=True, right_index=True)

    while i < n:
        newdf = pd.merge(newdf, lodf[i], left_index=True, right_index=True)

        labels.append(lodf[i].name)

        i += 1

    newdf.columns = labels

    return newdf


def rollingPCA(df, weights, pc=1):
    """

    Given a dataframe, window size (ws), and list of weights which is the same

    length as the window size, run a PCA. Output list of pc-tuples of

    eigenvectors for each day

    """

    ws = len(weights)

    newdf = logreturns(df)

    i = 0

    eigenvalues = [[] for i in range(pc)]

    while (i + ws) <= len(newdf):

        subdf = newdf.iloc[i:(i + ws), :]

        results = weightedPCA(subdf, weights, pc)

        for j in range(pc):
            eigenvalues[j].append(float(results[1][j]))

        i += 1

    f = pd.DataFrame(data=eigenvalues).T

    f.plot(figsize=(20, 7))


def weightedPCA(df, w, c):
    """

    Given a data frame (n=windowsize x m=numAssets), rescale the dataframe

    entries by using the weights (list of n scalars), and then perform regular

    PCA on this new scaled dataframe.

    """

    indx = range(len(w))

    newdf = pd.concat([w[i] * pd.Series.to_frame(df.iloc[i]).T for i in indx])

    results = dfPCA(newdf, c)

    return results


def dfPCA(df, n):
    """

    Given a dataframe, perform a PCA.

    Output a list of the first n eigenvalues.

    """

    pca = PCA(n_components=n)

    principalComponents = pca.fit(df)

    eivals = pca.explained_variance_ratio_

    return eivals


def logreturns(df):
    """

    Take a dataframe of price values, and output a dataframe of

    the daily returns.

    """

    logdf = df.applymap(math.log)

    differences = pd.DataFrame(data=np.diff(logdf, n=1, axis=0))

    #    newdf = np.divide(differences, logdf.drop(logdf.index.values[-1]))

    #    newdf.columns = logdf.columns.values

    #    newdf.index = logdf.index.values[:-1]

    #    return newdf

    differences.columns = logdf.columns.values

    differences.index = logdf.index.values[:-1]

    return differences


def expweight(lamda, n):
    """

    Create a list of weights, of length n. Each weight will be given by the

    exponential function with parameter lamda, but rescaled so that they

    sum to 1.

    """

    tempw = []

    total = 0

    i = 0

    while i < n:
        w = math.exp(lamda * i)

        tempw.append(w)

        total += w

        i += 1

    weights = [x / total for x in tempw]

    return weights







