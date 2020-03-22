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
from helpers import nn_arch, nn_reg, ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


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

    # clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    dim_red = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    dims_red_s = [2, 4, 6, 8, 10, 12, 14, 16]

    # %% data for 1

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    fs_madelon = rfc.fit(madelon_X, madelon_Y).feature_importances_
    fs_character = rfc.fit(character_X, character_Y).feature_importances_

    tmp = pd.Series(np.sort(fs_madelon)[::-1])
    tmp.to_csv(out + 'madelon scree.csv')

    tmp = pd.Series(np.sort(fs_character)[::-1])
    tmp.to_csv(out + 'character_scree.csv')

    # %% Data for 2
    filtr = ImportanceSelect(rfc)
    grid = {'filter__n': dim_red, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(madelon_X, madelon_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon dim red.csv')

    grid = {'filter__n': dims_red_s, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(character_X, character_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'character_dim_red.csv')
    #    raise
    # %% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 10
    filtr = ImportanceSelect(rfc, dim)

    madelon_X2 = filtr.fit_transform(madelon_X, madelon_Y)
    madelon_2 = pd.DataFrame(np.hstack((madelon_X2, np.atleast_2d(madelon_Y).T)))
    cols = list(range(madelon_2.shape[1]))
    cols[-1] = 'Class'
    madelon_2.columns = cols
    madelon_2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)

    dim = 10
    filtr = ImportanceSelect(rfc, dim)
    character_X2 = filtr.fit_transform(character_X, character_Y)
    character_2 = pd.DataFrame(np.hstack((character_X2, np.atleast_2d(character_Y).T)))
    cols = list(range(character_2.shape[1]))
    cols[-1] = 'Class'
    character_2.columns = cols
    character_2.to_hdf(out + 'datasets.hdf', 'character', complib='blosc', complevel=9)


if __name__ == '__main__':
    main()



