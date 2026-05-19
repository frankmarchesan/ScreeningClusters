# clustering.py
# the actual analysis bits: distance -> PCoA -> KMeans + hierarchical.
#
# main references I keep going back to:
#   sklearn clustering user guide:  https://scikit-learn.org/stable/modules/clustering.html
#   scipy distance functions:       https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
#   scipy hierarchy:                https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
#   PCoA explained (scikit-bio):    https://scikit-bio.org/docs/latest/generated/skbio.stats.ordination.pcoa.html
#   StatQuest on hierarchical:      https://www.youtube.com/watch?v=7xHsRkOdVwo
#   StatQuest on PCA/PCoA:          https://www.youtube.com/watch?v=FgakZw6K1QQ

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_distance(matrix):
    # Jaccard works well for binary "presence/absence" data.
    # pdist returns a condensed 1D array; squareform turns it into the NxN matrix.
    # docs -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    d = pdist(matrix.values, metric="jaccard")
    # NaNs can show up if two rows are both all-zero (won't happen here but just in case)
    d = np.nan_to_num(d, nan=0.0)
    return squareform(d)


def run_pcoa(dist_matrix, n_dims=3):
    # Classical PCoA = double-center the squared distances, then eigendecompose.
    #   https://www.davidzeleny.net/anadat-r/doku.php/en:pcoa_nmds
    # The math (Gower 1966) lives behind a paywall but the steps are:
    #   1. D2 = squared distances
    #   2. B  = -0.5 * H * D2 * H, where H = I - (1/n) * ones
    #   3. eigh(B) gives the coordinates
    # eigh docs -> https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html

    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist_matrix ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    # eigh returns ascending order, so flip them
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # only positive eigenvalues are "real" axes - negative ones come from the
    # distance not being perfectly Euclidean (Jaccard usually has a few).
    # TODO: should we warn if there are a lot of negative eigvals? for now just ignore.
    pos = eigvals > 0
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    # coordinates = eigenvectors scaled by sqrt(eigenvalue)
    coords = eigvecs[:, :n_dims] * np.sqrt(eigvals[:n_dims])

    # variance explained (only counts the positive eigvals)
    total = eigvals.sum()
    var_explained = (eigvals[:n_dims] / total) * 100
    return coords, var_explained


def run_kmeans(coords, k_min=2, k_max=8, random_state=42):
    # try a few values of k and pick the one with the best silhouette.
    # silhouette is between -1 and 1, higher = clusters are tighter and more separated.
    # explanation + plot:
    #   https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # KMeans docs:
    #   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    n = coords.shape[0]
    k_max = min(k_max, n - 1)
    ks = list(range(k_min, k_max + 1))

    inertias = []
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(coords)
        inertias.append(km.inertia_)
        # silhouette needs at least 2 distinct labels
        if len(set(labels)) > 1:
            sils.append(silhouette_score(coords, labels))
        else:
            sils.append(-1.0)

    best_k = ks[int(np.argmax(sils))]

    # refit at the chosen k to get the final assignments
    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    final_labels = km.fit_predict(coords) + 1  # +1 so clusters are 1-indexed, easier to read

    return final_labels, best_k, ks, inertias, sils


def run_hierarchical(dist_matrix, k, method="complete"):
    # Why complete linkage? single linkage tends to make weird "chained" clusters
    # where everything ends up in one big blob. complete linkage merges based on the
    # farthest pair so it gives more compact clusters.
    # quick read -> https://en.wikipedia.org/wiki/Complete-linkage_clustering
    # scipy linkage docs ->
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    #
    # linkage() wants a *condensed* distance vector (the upper triangle), not a square one.
    # squareform converts back and forth.
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    # fcluster gives us a flat assignment by cutting the tree to produce k groups
    # docs -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z
