# plotting.py
# matplotlib plotting helpers. nothing fancy, mostly copied from the docs/gallery.
#
# matplotlib gallery (where I get most of my plot ideas):
#   https://matplotlib.org/stable/gallery/index.html
# pyplot tutorial:
#   https://matplotlib.org/stable/tutorials/introductory/pyplot.html
# why we set the backend to "Agg" before importing pyplot:
#   so it works on machines without a display (e.g. when running headless)
#   https://matplotlib.org/stable/users/explain/figure/backends.html

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from scipy.cluster.hierarchy import dendrogram


def plot_pcoa_2d(coords, labels, var_explained, path):
    # standard 2D scatter colored by cluster.
    # scatter docs -> https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=30, alpha=0.85)
    ax.set_xlabel(f"PCo1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({var_explained[1]:.1f}%)")
    ax.set_title("PCoA - 2D (colors = clusters)")
    # legend by cluster id
    handles, _ = sc.legend_elements()
    ax.legend(handles, [f"Cluster {c}" for c in sorted(set(labels))], loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pcoa_3d(coords, labels, var_explained, path):
    # copied the pattern straight from the matplotlib 3D gallery:
    #   https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
    # the saved PNG is static - to actually rotate it you'd need to run plt.show() in a script
    # or use %matplotlib widget in a notebook.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=labels, cmap="tab10", s=30, alpha=0.85)
    ax.set_xlabel(f"PCo1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({var_explained[1]:.1f}%)")
    ax.set_zlabel(f"PCo3 ({var_explained[2]:.1f}%)")
    ax.set_title("PCoA - 3D")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_elbow_silhouette(ks, inertias, sils, best_k, path):
    # two side-by-side panels: inertia (the "elbow") and silhouette score.
    # elbow method explained -> https://en.wikipedia.org/wiki/Elbow_method_(clustering)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ks, inertias, "o-")
    ax1.axvline(best_k, color="red", linestyle="--", label=f"chosen k={best_k}")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia (sum of squared distances)")
    ax1.set_title("Elbow plot")
    ax1.legend()

    ax2.plot(ks, sils, "o-", color="green")
    ax2.axvline(best_k, color="red", linestyle="--", label=f"chosen k={best_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("silhouette score")
    ax2.set_title("Silhouette scores")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_pcoa_2d_subclusters(coords, parent_labels, sub_labels, var_explained, path):
#     parent cluster -> distinct hue from tab10.
#     subclusters within a parent -> shades (light to saturated) of that hue,
#     so parent grouping is readable at a glance AND subclusters within a parent
#     are still distinguishable AND no two parents share similar shades.
    parent_labels = np.asarray(parent_labels)
    sub_labels = np.asarray(sub_labels)
    base_cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for p_idx, p in enumerate(sorted(np.unique(parent_labels))):
        # pick base color from tab10 by parent index
        base = np.array(base_cmap(p_idx % 10)[:3])
        subs = sorted(np.unique(sub_labels[parent_labels == p]))
        for s_idx, s in enumerate(subs):
            # shade factor t in [0.4, 1.0]: t=1 -> base color, t<1 -> mixed with white
            t = 1.0 if len(subs) == 1 else 0.4 + 0.6 * (s_idx / (len(subs) - 1))
            color = tuple(base * t + (1 - t))
            mask = (parent_labels == p) & (sub_labels == s)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=color, s=30, alpha=0.85,
                       label=f"Cluster {p}.{s}")
    ax.set_xlabel(f"PCo1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({var_explained[1]:.1f}%)")
    ax.set_title("PCoA - 2D (parent = hue, subcluster = shade)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_pcoa_k_grid(coords, k_to_labels, var_explained, path, ncols=4):
    # one figure, one panel per k (k_to_labels maps k -> cluster labels).
    # lays the panels out in a grid so all the k values can be compared side by side
    # instead of flipping between separate PNGs.
    # subplots grid -> https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    # TODO: fill body
    raise NotImplementedError


def plot_dendrogram(linkage_matrix, path, truncate=30):
    # the full dendrogram is unreadable with 200+ participants,
    # so truncate_mode="lastp" only shows the last `truncate` merges.
    # dendrogram docs ->
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix,
               truncate_mode="lastp",
               p=truncate,
               leaf_rotation=90,
               show_leaf_counts=True,
               ax=ax)
    ax.set_title(f"Hierarchical clustering dendrogram (last {truncate} merges)")
    ax.set_xlabel("cluster size (in parens) or sample id")
    ax.set_ylabel("distance")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
