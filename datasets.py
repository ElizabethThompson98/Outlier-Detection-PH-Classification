#!/usr/bin/env python3

import csv
from typing import Optional
from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt



def sample_from_circle(
        n_pts: int = 100,
        variation: float = 0.1,
        outlier: bool = False,
        seed: Optional[int] = None,
        ):
    if outlier is True:
        n_pts = n_pts - 1

    rng = np.random.default_rng(seed)

    r = 1
    t = np.linspace(0, 2*np.pi * (n_pts-1)/n_pts, n_pts)
    x = r*np.cos(t) + variation * rng.random(n_pts)
    y = r*np.sin(t) + variation * rng.random(n_pts)
    output = np.vstack((x,y)).transpose()

    if outlier is True:
        output = np.append(output,[[0,0]],axis=0)

    return output



def sample_from_ellipse(
        n_pts: int = 100,
        axis_1: float = 2,
        axis_2: float = 1,
        variation: float = 0.1,
        phase: float = 0,
        ):
    t = np.linspace(0, 2*np.pi * (n_pts-1)/n_pts, n_pts) + phase
    x = axis_1 * np.cos(t) + variation * np.random.rand(n_pts)
    y = axis_2 * np.sin(t) + variation * np.random.rand(n_pts)
    return np.vstack((x,y)).transpose()



def sample_from_cassini_oval(
        n_pts: int,
        variation: float = 0.1
        ):
    t = np.linspace(-1, 1, int(n_pts/2))
    x = np.concatenate((t,t)) + variation * np.random.rand(n_pts)
    yh = (t**2 + 0.5) * np.sqrt(1 - t**2)
    y = np.concatenate((-yh, yh)) + variation * np.random.rand(n_pts)

    return np.vstack((x,y)).transpose()



def sample_from_sphere(
        n_pts: int,
        ambient_dim: int = 3,
        radius: float = 1,
        ):
    vec = np.random.randn(ambient_dim, n_pts)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    return vec.transpose()



def sample_from_torus(
        n_pts: int,
        R: float = 2,
        r: float = 1
        ):
    nPtsSampled = 0
    theta = np.zeros([n_pts])
    phi = np.zeros([n_pts])

    # rejection sampling
    while nPtsSampled < n_pts:
        thetaSample = 2 * np.pi * np.random.rand()
        phiSample = 2 * np.pi * np.random.rand()
        W = 2 * np.pi * np.random.rand()

        if W <= (R + r * np.cos(thetaSample))/(R+r):
            theta[nPtsSampled] = thetaSample
            phi[nPtsSampled] = phiSample
            nPtsSampled += 1


        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)

    return np.vstack((x,y,z)).transpose()



def sample_from_figure_eight(
        n_pts: int,
        scaling: float = 1,
        neck_size: float = 0.5,
        variation: float = 0,
        seed: Optional[int] = None
        ):
    # adapted from Bastian Rieck
    rng = np.random.default_rng(seed)

    T = np.linspace(-np.pi, np.pi, num=n_pts, endpoint=False)
    X = scaling * np.sin(T)
    Y = scaling * np.sin(T)**2 * np.cos(T) + neck_size * np.cos(T)

    X = np.column_stack((X, Y))
    X += rng.uniform(0, variation, size=(n_pts, 2))
    return X



def sample_from_annulus(
        n_pts: int,
        r: float = 1,
        R: float = 2,
        seed: Optional[int] = None
):
    # adapted from Bastian Rieck
    if r >= R:
        raise RuntimeError(
            'Inner radius must be less than or equal to outer radius'
        )

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, n_pts)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(rng.uniform(r ** 2, R ** 2, n_pts))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X




def sample_from_disc(
        n_pts: int,
        radius: float = 1.0,
        center: tuple = (0.0, 0.0),
        seed: Optional[int] = None
):

    rng = np.random.default_rng(seed)

    # Sample radius and angle
    r = radius * np.sqrt(rng.uniform(0, 1, size=n_pts))  # sqrt ensures uniformity in area
    theta = rng.uniform(0, 2 * np.pi, size=n_pts)

    # Convert to Cartesian coordinates
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack((x, y))



def point_in_polygon(
        random_points,
        polygon_points
):
    """
    Vectorized even-odd (ray casting) test.
    pts:  (N,2) array of points
    poly: (M,2) array of polygon vertices in order (closed or open)
    returns: (N,) boolean mask: True if inside
    """
    x, y = random_points[:, 0], random_points[:, 1]
    xv = np.asarray(polygon_points)[:, 0]
    yv = np.asarray(polygon_points)[:, 1]

    # Edges: (xk,yk) -> (xj,yj)
    xk, yk = xv, yv
    xj, yj = np.roll(xv, -1), np.roll(yv, -1)

    # Broadcast points against edges
    y = y[:, None]
    x = x[:, None]

    # Check if edge straddles the horizontal ray from point
    straddles = ( (yk > y) != (yj > y) )

    # X coordinate of intersection of edge with the ray at each y
    x_intersect = (xj - xk) * ( (y - yk) / (yj - yk) ) + xk

    crossings = straddles & (x < x_intersect)
    inside = crossings.sum(axis=1) % 2 == 1
    return inside



def sample_points_in_polygon(
        poly: np.array,
        n_pts: int,
        batch: int = 10000,
        seed: Optional[int] = None,
):
    """
    Uniform samples inside a simple polygon (no holes) using rejection sampling.
    poly: (M,2) vertices
    """
    rng = np.random.default_rng(seed)
    poly = np.asarray(poly)
    minx, miny = poly.min(axis=0)
    maxx, maxy = poly.max(axis=0)

    out = []
    total = 0
    while total < n_pts:
        k = min(batch, n_pts - total)
        xs = rng.uniform(minx, maxx, size=k)
        ys = rng.uniform(miny, maxy, size=k)
        pts = np.column_stack([xs, ys])
        mask = point_in_polygon(pts, poly)  # boolean mask of length k
        if mask.any():
            acc = pts[mask]
            out.append(acc)
            total += acc.shape[0]

    return np.vstack(out)[:n_pts]



def fill_curve(
        curve_points,
        n_pts: int = 100,
        seed: Optional[int] = None,
        ):
    return sample_points_in_polygon(poly=curve_points, n_pts = n_pts, seed=seed)




def compute_outliers(
        points: np.ndarray,
        n_outliers: int = 1,
        min_distance_factor: float = 2,
        max_distance_factor: float = 4,
        seed: Optional[int] = None,
):
    # Computes outliers for the point cloud `points`. The outliers are away from the point
    # cloud, at a distance specified by min_distance_factor and max_distance_factor.

    rng = np.random.default_rng(seed)
    outliers = []

    for _ in range(n_outliers):
        center_of_mass = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center_of_mass, axis=1)
        furthest_point_norm = np.max(distances)

        outlier_norm = rng.uniform(min_distance_factor, max_distance_factor) * furthest_point_norm
        phi = rng.random()*2*np.pi
        outlier = outlier_norm * np.array([np.cos(phi), np.sin(phi)])
        # outliers = np.vstack([outliers, outlier])
        outliers.append(outlier)

    return np.array(outliers)



def generate_spaced_points(
        n_pts: int,
        min_distance_between_points: float = 2,
        bounds: list[float] = [-10,10],
        seed: Optional[int] = None
):
    # Generates n_pts spaced points, at least min_distance_between_points
    # away from each other and all living within a cube with sides given by bounds.
    # The idea is to use this function to generate centers of clusters.

    rng = np.random.default_rng(seed)
    points = []

    max_attempts = 1000 * n_pts
    attempts = 0

    while len(points) < n_pts and attempts < max_attempts:
        candidate = rng.uniform(bounds[0], bounds[1], size=2)
        if all(np.linalg.norm(candidate - p) >= min_distance_between_points for p in points):
            points.append(candidate)
        attempts += 1

    if len(points) < n_pts:
        raise ValueError("Could not place all points with the given spacing.")

    return np.array(points)



def divide_per_classes(
        n: int,
        n_classes: int,
        min_per_class = 0,
        seed: Optional[int] = None,
):
    # returns a list of `n_classes` non-negative elements that are all greater than or
    # equal to min_per_class and that add up to n
    if n < min_per_class * n_classes:
        raise ValueError(f"Cannot divide {n} elements into {n_classes} classes \
        with at least 3 per class.")

    rng = np.random.default_rng(seed)
    class_sizes = [min_per_class] * n_classes
    remaining = n - min_per_class * n_classes

    for _ in range(remaining):
        class_sizes[rng.integers(n_classes)] += 1

    return class_sizes



def plot_points(
        points: np.ndarray
):
    plt.scatter(points[:,0], points[:,1], s=0.5)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()



@dataclass
class Cluster():
    points: np.ndarray
    label: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "points": self.points.tolist()
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            points=np.array(d["points"]),
            label=d["label"]
        )


def flatten_cluster_list(
        clusters: list[Cluster]
) -> np.ndarray:
    points = np.empty([1,2])
    for cluster in clusters:
        points = np.append(points,cluster.points, axis=0)
    return points



def move_clusters(
        clusters: list[Cluster],
        centers: np.ndarray
        ):
    for i, cluster in enumerate(clusters):
        cluster.points += centers[i]
    return clusters



def read_points_from_csv(
        filename: str,
        ):
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        points = [[float(row["x"]), float(row["y"])] for row in reader]

    return np.array(points)



def main():

    # --- Parameters ---
    n_cluster_pts = 100
    n_outliers = 3
    seed = 0
    filename_clusters = "data/four_clusters.json"

    # --- Adding clusters (centered at 0) ---
    clusters = [
        Cluster(points = sample_from_circle(n_cluster_pts, seed=seed), label = "0"),
        Cluster(points = sample_from_disc(2*n_cluster_pts, seed=seed), label = "1"),
    ]
    clusters.append(Cluster(points=4*read_points_from_csv("data/custom_outline_0.csv"), label="2"))
    filled_curve_points = fill_curve(read_points_from_csv("data/custom_outline_1.csv"), n_pts=n_cluster_pts, seed=seed)
    clusters.append(Cluster(points = 2*filled_curve_points, label = "3"))


    # --- Moving clusters around ---
    cluster_centers = generate_spaced_points(len(clusters), seed=seed)
    clusters = move_clusters(clusters, cluster_centers)


    # --- Adding outliers ---
    n_outliers_per_cluster = divide_per_classes(n_outliers, len(clusters))
    outliers = []
    for i, cluster in enumerate(clusters):
        outlier_for_cluster = compute_outliers(cluster.points, n_outliers_per_cluster[i])
        if outlier_for_cluster.size > 0:
            outliers.append(outlier_for_cluster)
    outliers = np.vstack(outliers)
    clusters.append(Cluster(points=outliers, label="outliers"))


    points = flatten_cluster_list(clusters)
    print(f"Generated a total of {len(points)} points.")


    # --- Saving points to JSON ---
    with open(filename_clusters, "w") as f:
        json.dump([c.to_dict() for c in clusters], f)
    print(f"Points saved to {filename_clusters}")


    # --- Plotting points ---
    plot_points(points)



if __name__ == "__main__":
    main()
