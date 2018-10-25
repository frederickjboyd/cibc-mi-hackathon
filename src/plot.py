import matplotlib.pyplot as plt
import numpy as np


def plotTest(x, y, xlabel, ylabel):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
