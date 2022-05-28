import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os
import numpy as np
import matplotlib.pyplot as plt

with open('data/x.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/y.pickle', 'rb') as handle:
    y = pickle.load(handle)


os.makedirs( 'figs', exist_ok=True )
os.makedirs( 'figs/tests', exist_ok=True )
save_folder = 'figs/tests/'

for i in range(10):
    for j in range(10):
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(x[i*100 + j,:])
        plt.subplot(2,1,2)
        plt.imshow(y[i*100 + j,:,:])
        plt.savefig( save_folder + 't_'+str(i*100 + j)+'.png', dpi=300 )
