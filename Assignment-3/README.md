# Unsupervised-Machine_learning

### Datasets

* Madelon - http://archive.ics.uci.edu/ml/datasets/madelon
* Letter Recognition - https://archive.ics.uci.edu/ml/datasets/letter+recognition



### Below are the steps to run the code.

1) Create a BASE folder in your DIR and put both the datasets in it and run createHDF.py to create the .hdf files.
2) Run clusters.py pointing to BASE dir for both datasets.
3) Run PCA.py, ICA.py, RP.py and RF.py pointing to BASE dir for both datasets and output to respective dir.
4) Run clustering.py pointing to PCA/ICA/RF/RP dir Part 3. of assignment using Madelon and Letter Recognition dataset.hdf.
5) Run neuralNetwork.py pointing to PCA/ICA/RF/RP dir for Part 4. of assignment using Madelon dataset.hdf.
6) Run clusters.py pointing to BASE dir , followed by neuralNetwork.py for Part 5. of assignment using Madelon dataset.hdf.
7) Use plot_me.py, silhouette.py and BIC.py for generating plots.

### Dependencies
Python, Scikitlearn Library, Pandas, Matplotlib


### References
* http://fourier.eng.hmc.edu/e161/lectures/ica/node4.html
* https://github.com/JonathanTay/CS-7641-assignment-3
