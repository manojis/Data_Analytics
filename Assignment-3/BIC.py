import numpy as np
import itertools
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as pl
import matplotlib as mpl
from sklearn.mixture import GaussianMixture as GMM
from sklearn import mixture
from sklearn.preprocessing import StandardScaler


"""

Plot AIC/BIC curves

"""
#source - https://gist.github.com/jakevdp/1534486/9380b656663afbeb8690dbcde28dcc3329168638

base = './BASE/'  ## substitute the desired dir here

np.random.seed(0)

letter = pd.read_hdf(base + 'datasets.hdf', 'letter')
X = letter.drop('Class', 1).copy().values
letter_Y = letter['Class'].copy().values

# madelon = pd.read_hdf(base + 'datasets.hdf', 'madelon')
# X = madelon.drop('Class', 1).copy().values
# madelon_Y = madelon['Class'].copy().values

np.random.seed(0)
#madelon_X = StandardScaler().fit_transform(X)
letter_X = StandardScaler().fit_transform(X)

n_components = np.array([2, 5, 10, 15, 20, 25, 30, 35, 40])

BIC = np.zeros(n_components.shape)
AIC = np.zeros(n_components.shape)

for i, n in enumerate(n_components):
    clf = GMM(n_components=n,
              covariance_type='diag')
    clf.fit(X)

    AIC[i] = clf.aic(X)
    BIC[i] = clf.bic(X)

pl.figure()
pl.plot(n_components, AIC, label='AIC')
pl.plot(n_components, BIC, label='BIC')
pl.legend(loc=0)
pl.xlabel('n_components')
pl.ylabel('AIC / BIC')
pl.title("AIC/BIC Letter")
pl.show()