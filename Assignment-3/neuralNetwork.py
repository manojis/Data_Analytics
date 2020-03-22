
"""
@author: ssunitha3

"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from util import plot_learning_curve,plot_validation_curve
from util import import_data
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time as time

"""
Assignment 1 Neural Network 
"""

def main():


        base = './PCA/' # change the DIR accordingly
        np.random.seed(0)

        madelon = pd.read_hdf(base + 'datasets.hdf', 'madelon')
        madelon_X = madelon.drop('Class', 1).copy().values
        madelon_Y = madelon['Class'].copy().values.reshape(-1,1)

        X_train, X_test, y_train, y_test = train_test_split(madelon_X, madelon_Y, test_size=0.3, random_state=42)

        np.random.seed(0)

        mlp = MLPClassifier(activation='relu',verbose=10, max_iter=2000, early_stopping=True, random_state=5,alpha=10 ** -5,hidden_layer_sizes=(62,62))
        tick = time.clock()
        mlp.fit(X_train, y_train)
        tock = time.clock() - tick
        print("Traning time for {} dataset".format("Madelon"), tock)
        tick = time.clock()
        y_pred = mlp.predict(X_test)
        tock = time.clock() - tick
        print("Testing time for {} dataset".format("Madelon"), tock)
        print("Accuracy Score for Madelon", accuracy_score(y_test, y_pred))

        ##Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        sns.heatmap(cm, center=True)
        # plt.show()
        plt.savefig('ConfusionMatrix for Madelon')



if __name__ == '__main__':

   main()