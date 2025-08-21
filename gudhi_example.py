#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
import gudhi as gd

from datasets import Cluster, flatten_cluster_list



def main():
    filename_clusters = "data/four_clusters.json"
    with open(filename_clusters, "r") as f:
        data = json.load(f)

    clusters = [Cluster.from_dict(d) for d in data]
    points = flatten_cluster_list(clusters)

    print(f"Read in {len(points)} points from {filename_clusters}.")

    rips_complex = gd.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    barcode = simplex_tree.persistence()

    gd.plot_persistence_barcode(barcode)
    plt.show()



if __name__ == "__main__":
    main()
