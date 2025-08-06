#Received from multipers documentation: I'm troubleshooting
#getting the multipers package to be installed correctly right now

#imports
import multipers as mp
import multipers.filtrations as mpf
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np

#generate point cloud
from multipers.data.synthetic import noisy_annulus
np.random.seed(0)
X = noisy_annulus(200, 0)
plt.scatter(*X.T)
plt.gca().set_aspect(1)

#compute simplex tree of point cloud
st = gd.RipsComplex(points = X).create_simplex_tree(max_dimension=2)

#run Vietoris-Rips filtration & compute barcode using simplex tree
st.compute_persistence()
pers = st.persistence_intervals_in_dimension(1)
gd.plot_persistence_barcode(pers)
