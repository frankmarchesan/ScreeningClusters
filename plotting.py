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
