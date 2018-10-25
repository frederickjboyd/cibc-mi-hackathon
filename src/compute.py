import pandas as pd
from scipy import stats
import numpy as numpy


def averageProcedurePrice(df):
    # Extract procedure code and prices column
    extractedMatrix = df.iloc[:, 6:]
    # Calculate mean price of each procedure and return it
    return extractedMatrix.groupby([6]).mean()


def averageStatePrice(df):
    # Extract state code, prices column
    extractedMatrix = df.iloc[:, [4, 7]]
    # Calculate mean price for each state and return it
    return extractedMatrix.groupby([4]).mean()


def zScoreByCategory(df, category):
    # Extract category, prices column
    extractedMatrix = df.iloc[:, [category, 7]]

    # Get groups of categories
    grouped = extractedMatrix.groupby([category])

    # Create numpy array
    zscore = numpy.zeros(df.shape[0])

    for name, group in grouped:
        indices = group.index.values
        zsc = stats.zscore(group[7])

        for i in range(0, len(zsc)):
            if numpy.isnan(zsc[i]):
                zscore[indices[i]] = 0
            else:
                zscore[indices[i]] = zsc[i]

    return zscore
