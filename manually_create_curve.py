#!/usr/bin/env python3

import csv

import numpy as np
import matplotlib.pyplot as plt



def draw_and_sample_line(n_pts=20, csv_path="data/custom_outline.csv"):
    """
    Interactive GUI: user draws a line with the mouse.
    On mouse release, the line is resampled at n_pts equally spaced points.

    Controls:
      - Left mouse: draw a stroke
      - Release mouse: show sampled points
      - r: reset/clear the canvas
      - w: write current sampled points to CSV (csv_path)

    Args:
        n_pts (int): Number of equally spaced sample points.
        csv_path (str): Path to CSV file for saving.

    Returns:
        points (list of tuple): The last sampled (x, y) coordinates.
    """
    fig, ax = plt.subplots()
    ax.set_title("Draw with left mouse. Release to sample.\n'r' to reset, 'w' to write to csv.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    drawing = {"active": False, "xs": [], "ys": []}
    (line_plot,) = ax.plot([], [], lw=2)
    sample_scat = ax.scatter([], [], s=30, zorder=3)
    sampled_points = []

    def _resample_polyline(x, y, n):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            return x, y
        dx = np.diff(x)
        dy = np.diff(y)
        seg = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] == 0:
            return np.array([x[0]] * n), np.array([y[0]] * n)
        s_target = np.linspace(0, s[-1], n)
        x_eq = np.interp(s_target, s, x)
        y_eq = np.interp(s_target, s, y)
        return x_eq, y_eq

    def on_press(event):
        if event.button != 1 or event.inaxes != ax:
            return
        drawing["active"] = True
        drawing["xs"], drawing["ys"] = [], []
        if event.xdata is not None and event.ydata is not None:
            drawing["xs"].append(event.xdata)
            drawing["ys"].append(event.ydata)
        line_plot.set_data(drawing["xs"], drawing["ys"])
        fig.canvas.draw_idle()

    def on_move(event):
        if not drawing["active"] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        drawing["xs"].append(event.xdata)
        drawing["ys"].append(event.ydata)
        line_plot.set_data(drawing["xs"], drawing["ys"])
        fig.canvas.draw_idle()

    def on_release(event):
        nonlocal sampled_points
        if event.button != 1:
            return
        drawing["active"] = False
        if len(drawing["xs"]) < 2:
            return
        x_eq, y_eq = _resample_polyline(drawing["xs"], drawing["ys"], n_pts)
        sampled_points = list(zip(x_eq, y_eq))
        sample_scat.set_offsets(np.c_[x_eq, y_eq])
        fig.canvas.draw_idle()

        print(f"Sampled {len(sampled_points)} points")

    def on_key(event):
        nonlocal sampled_points
        if event.key == "r":  # reset
            drawing["xs"].clear()
            drawing["ys"].clear()
            line_plot.set_data([], [])
            sample_scat.set_offsets(np.empty((0, 2)))
            sampled_points = []
            print("Canvas cleared.")
            fig.canvas.draw_idle()

        elif event.key == "w":  # save points
            if not sampled_points:
                print("No points to save yet!")
                return
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(sampled_points)
            print(f"Saved {len(sampled_points)} points to {csv_path}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return sampled_points



if __name__ == "__main__":

    n_pts = 20
    pts = draw_and_sample_line(n_pts)
