# Python program to demonstrate 
# HDF5 file 
  
import numpy as np 
import h5py 
  
# initializing a random numpy array 
arr = np.random.randn(1000) 
  
# creating a file 
with h5py.File('datasets.hdf', 'r') as f:  
    data = f['default'] 
    # get the values ranging from index 0 to 15 
    print(data[:150]) 
