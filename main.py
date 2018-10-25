from src import filesys, compute, plot
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.neighbors import LocalOutlierFactor

# Directory where data is located
#dataDirectory = "./sample100000.csv"
dataDirectory = "./sample1000.csv"

# Initialize the files class
df = filesys.files(dataDirectory)
print (df.getData())
#averageProcedurePrice = compute.averageProcedurePrice(df.getData())
#averageStatePrice = compute.averageStatePrice(df.getData())

zScores = compute.zScoreByCategory(df.getData(), 4)
zS = compute.zScoreByCategory(df.getData(), 6)
array = np.column_stack((zScores, zS))
# print(type(zScores))
# print(type(zS))
print(array)

#plot.plotTest(zScores, zS, 'State', 'Procedure Type')
params = {'quantile': .3,
          'eps': .3,
          'damping': .9,
          'preference': -200,
          'n_neighbors': 10,
          'n_clusters': 3}

# connectivity matrix for structured Ward
# connectivity = kneighbors_graph(
#    array, n_neighbors=params['n_neighbors'], include_self=False)
# make connectivity symmetric
# connectivity = 0.5 * (connectivity + connectivity.T)

# algorithms
# two_means = cluster.MiniBatchKMeans(n_clusters=5)

clf = LocalOutlierFactor(n_neighbors=10)

y_pred = clf.fit_predict(array)
outliers = y_pred[200:]
vals = clf.negative_outlier_factor_

# print (y_pred)
# print(vals)

dist = list()
for each in array:
    dist.append(np.power(each[0], 2) + np.power(each[1], 2))

npList = np.column_stack(
    (df.getData().index.values, df.getData().iloc[:, 0:], dist))

print(npList)

# npList.sort(2)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))
plt.scatter(array[:, 0], array[:, 1], s=10, color=colors[y_pred])

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())

plt.show()
