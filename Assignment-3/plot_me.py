import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns; sns.set()


"""
@author santhanu

To plot the various curves in Assignment 3 cs7641

"""

#SSE Curve

input = pd.read_csv('./OutputPostRF/SSE.csv', index_col='Clusters')

input = input.drop(input.columns[0],axis=1)

input.plot()
plt.title("Letter SSE vs No. of Clusters")

plt.ylabel("SSE")
plt.tight_layout()
plt.show()
plt.clf()
#

# AMI Curve

ami = pd.read_csv('./Output/letter adjMI.csv', index_col='Clusters')

input = ami.drop(ami.columns[1],axis=1)

input.plot()
plt.title("Letter K means AMI vs No. of Clusters")

plt.ylabel("AMI")
plt.tight_layout()
plt.show()
plt.clf()
#

#Cluster Accuracy Curve

input = pd.read_csv('./OutputPostRF/letter acc.csv', index_col='Clusters')

input.plot()
plt.title("Letter Kmeans & GMM  Accuracy vs No. of Clusters")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
plt.clf()

#
#Loglikelihood Curve

acc = pd.read_csv('./OutputPostRF/logliklihood.csv', index_col='Clusters')

input = acc.drop(acc.columns[1],axis=1)

input.plot()
plt.title("Madelon Logliklihood vs No. of Clusters")

plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

plt.clf()
#

#PCA 2 D projection Plot
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
# # import some data to play with
base = './BASE/'
#
letter = pd.read_hdf(base + 'datasets.hdf', 'letter')
enc = OrdinalEncoder()
X = letter.drop('Class', 1).copy().values
Y = letter['Class'].copy().values.reshape(-1,1)
y = enc.fit_transform(Y)
letter_new = np.hstack((X,y))

madelon = pd.read_hdf(base + 'datasets.hdf', 'madelon')
X = madelon.drop('Class', 1).copy().values
y = madelon['Class'].copy().values

X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
#
min = X[:, 0].copy()
min = pd.to_numeric(min)

max =  X[:, 0].copy()
max = pd.to_numeric(max)

x_min, x_max = min.min() - .5, max.max() + .5
y_min, y_max = min.min() - .5, max.max() + .5


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

#Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(madelon)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


projected = PCA(n_components=5).fit_transform(letter_new)

plt.scatter(projected[:, 0], projected[:, 1],
            c=y.ravel(), edgecolor='none', alpha=1,
            cmap='tab20c')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.title("Letter PCA component 1 vs component 2")


#Pairwise Distance Curve

data = pd.read_csv('./RP/letter scree1.csv', index_col='No. of Dimensions')

input = data.drop(data.columns[1],axis=1)

data.plot()
plt.title("Letter RP Pairwise Distance Variance")

plt.ylabel("Pairwise Variance")
plt.tight_layout()
plt.show()

plt.clf()
#

#Reconstruction Error Curve

data = pd.read_csv('./RP/letter scree2.csv', index_col='No. of Dimensions')

#input = data.drop(data.columns[1],axis=1)

data.plot()
plt.title("Letter RP Reconstruction Error Variance")

plt.ylabel("Recontruction Error")
plt.tight_layout()
plt.show()

plt.clf()


#
#Homogeneity Curve


input = pd.read_csv('./Output/cs.csv', index_col='Clusters')

input = input.drop(input.columns[0],axis=1)

input.plot()
plt.title("Madelon Homogeneity Score")

plt.ylabel("Score")
plt.tight_layout()
plt.show()

plt.clf()


#
#Completeness Curve

input = pd.read_csv('./Output/hs.csv', index_col='Clusters')

input = input.drop(input.columns[1],axis=1)

input.plot()
plt.title("Letter Completeness Score")

plt.ylabel("Score")
plt.tight_layout()
plt.show()

plt.clf()





