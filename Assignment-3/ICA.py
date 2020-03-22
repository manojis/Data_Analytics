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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA


def main():
    out = './BASES/'
    np.random.seed(0)
    character = pd.read_hdf('./BASES/datasets.hdf', 'character')
    character_X = character.drop('Class', 1).copy().values
    character_Y = character['Class'].copy().values

    madelon = pd.read_hdf('./BASES/datasets.hdf', 'madelon')
    madelon_X = madelon.drop('Class', 1).copy().values
    madelon_Y = madelon['Class'].copy().values

    madelon_X = StandardScaler().fit_transform(madelon_X)
    character_X = StandardScaler().fit_transform(character_X)

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    dim_red = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    dims_red_s = [2, 4, 6, 8, 10, 12, 14, 16]

    # raise data for 1
    ################################
    ica = FastICA(random_state=5)
    kurt = {}
    for dim in dims_red_s:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(character_X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt)
    kurt.to_csv(out + 'character_scree.csv')
    ################################
    ica = FastICA(random_state=5)
    kurt = {}
    for dim in dim_red:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(madelon_X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt)
    kurt.to_csv(out + 'madelon_scree.csv')
    raise

    # Data for 2
    ##############################

    grid = {'ica__n_components': dims_red_s, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    ica = FastICA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('ica', ica), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(character_X, character_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'character_dim_red.csv')
    ##############################
    grid = {'ica__n_components': dim_red, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    ica = FastICA(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('ica', ica), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(madelon_X, madelon_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon_dim_red.csv')

    # raise data for 3
    ###############################
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 16
    ica = FastICA(n_components=dim, random_state=10)
    character_X2 = ica.fit_transform(character_X)
    character_2 = pd.DataFrame(np.hstack((character_X2, np.atleast_2d(character_Y).T)))
    cols = list(range(character_2.shape[1]))
    cols[-1] = 'Class'
    character_2.columns = cols
    character_2.to_hdf(out + 'datasets.hdf', 'character', complib='blosc', complevel=9)

    #################################
    dim = 45
    ica = FastICA(n_components=dim, random_state=10)
    madelon_X2 = ica.fit_transform(madelon_X)
    madelon_2 = pd.DataFrame(np.hstack((madelon_X2, np.atleast_2d(madelon_Y).T)))
    cols = list(range(madelon_2.shape[1]))
    cols[-1] = 'Class'
    madelon_2.columns = cols
    madelon_2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)


if __name__ == '__main__':
    main()
