"""
    @author Manoj.Mohanan Nair
    @Date 03/14/20
    @Description: Dimensionality Reduction with PCA technique
    @Reference: https://github.com/JonathanTay/CS-7641-assignment-3
                which was used as a reference in creating my own code.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


def main():
    out = './BASE/'
    cmap = cm.get_cmap('Spectral')

    madelon = pd.read_hdf('./BASE/datasets.hdf', 'madelon')
    madelon_X = madelon.drop('Class', 1).copy().values
    madelon_Y = madelon['Class'].copy().values
    madelon_X = StandardScaler().fit_transform(madelon_X)

    np.random.seed(0)
    character = pd.read_hdf('./BASE/datasets.hdf', 'character')
    character_X = character.drop('Class', 1).copy().values
    character_Y = character['Class'].copy().values
    character_X = StandardScaler().fit_transform(character_X)

    dim_red = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    dims_red_s = [2, 4, 6, 8, 10, 12, 14, 16]

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    # 5, 10, raise
    # %% data for 1
    pca = PCA(random_state=5)
    pca.fit(character_X)
    tmp = pd.Series(data=pca.explained_variance_, index=range(1, 17))
    tmp.to_csv(out + 'characterPCA.csv')

    pca = PCA(random_state=5)
    pca.fit(madelon_X)
    tmp = pd.Series(data=pca.explained_variance_, index=range(1, 501))
    tmp.to_csv(out + 'madelonPCA.csv')
    ########################################
    # %% Data for 2
    grid = {'pca__n_components': dims_red_s, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('pca', pca), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(character_X, character_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'character_dim_red.csv')

    grid = {'pca__n_components': dim_red, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('pca', pca), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(madelon_X, madelon_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon_dim_red.csv')
    #########################################
    # raise
    # %% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 14
    pca = PCA(n_components=dim, random_state=10)
    character_X2 = pca.fit_transform(character_X)
    character_2 = pd.DataFrame(np.hstack((character_X2, np.atleast_2d(character_X).T)))
    cols = list(range(character_2.shape[1]))
    cols[-1] = 'Class'
    character_2.columns = cols
    character_2.to_hdf(out + 'datasets.hdf', 'character', complib='blosc', complevel=9)

    dim = 5
    pca = PCA(n_components=dim, random_state=10)
    madelon_X2 = pca.fit_transform(madelon_X)
    madelon_2 = pd.DataFrame(np.hstack((madelon_X2, np.atleast_2d(madelon_Y).T)))
    cols = list(range(madelon_2.shape[1]))
    cols[-1] = 'Class'
    madelon_2.columns = cols
    madelon_2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)


if __name__ == '__main__':
    main()
