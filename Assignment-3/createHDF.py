"""
    @author Manoj.Mohanan Nair
    @Date 03/12/20
    @Reference: https://www.datacamp.com/
"""

import pandas as pd
import numpy as np
import os
import sklearn.model_selection as ms

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './BASE/'

character = pd.read_csv("LetterRecognition.csv",header=None)
character = character.iloc[1:]

cols = list(range(character.shape[1]))
cols[-1] = 'Class'
character.columns = cols
character.to_hdf(OUT+'datasets.hdf', 'character', complib='blosc', complevel=9)

mad_X1 = pd.read_csv('madelon_train.data', header=None, sep=' ')
mad_X2 = pd.read_csv('madelon_valid.data', header=None, sep=' ')
mad_Y1 = pd.read_csv('madelon_train.labels', header=None, sep=' ')
mad_Y2 = pd.read_csv('madelon_valid.labels', header=None, sep=' ')

mad_X = pd.concat([mad_X1, mad_X2], 0).astype(float)
mad_Y = pd.concat([mad_Y1, mad_Y2], 0)
mad_Y.columns = ['Class']

madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(
    mad_X, mad_Y, test_size=0.3, random_state=0, stratify=mad_Y)

mad_X = pd.DataFrame(madelon_trgX)
mad_Y = pd.DataFrame(madelon_trgY)
mad_Y.columns = ['Class']

mad_X2 = pd.DataFrame(madelon_tstX)
mad_Y2 = pd.DataFrame(madelon_tstY)
mad_Y2.columns = ['Class']

mad1 = pd.concat([mad_X, mad_Y], 1)
mad1 = mad1.dropna(axis=1, how='all')
mad2 = pd.concat([mad_X2, mad_Y2], 1)
mad2 = mad2.dropna(axis=1, how='all')

mad1.to_hdf(OUT+'datasets.hdf', 'madelon', complib='blosc', complevel=9)
mad2.to_hdf(OUT+'datasets.hdf', 'madelon_test', complib='blosc', complevel=9)