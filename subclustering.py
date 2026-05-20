# subclustering.py
# post-clustering analysis: sub-cluster within parent clusters, and
# summarise the top taxonomy codes per (sub)cluster so patterns are readable.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans_sub(coords, parent_labels, sub_k, random_state=42):
    # re-run KMeans within each parent cluster on the same PCoA coords.
    # returns 1D array of sub-cluster labels (1-indexed *per parent*).
    parent_labels = np.asarray(parent_labels)
    sub = np.ones(len(parent_labels), dtype=int)
    for c in np.unique(parent_labels):
        idx = np.where(parent_labels == c)[0]
        if len(idx) < sub_k or len(idx) < 2:
            print(f"  cluster {c}: only {len(idx)} members, skipping sub-clustering")
            continue
        km = KMeans(n_clusters=sub_k, n_init=10, random_state=random_state)
        sub[idx] = km.fit_predict(coords[idx]) + 1
    return sub


def top_codes(matrix, labels, top_n=10):
    # matrix: DataFrame (participants x codes), labels: array aligned to matrix.index.
    # returns long-form DataFrame: Group, Code, Count, Proportion, Group Size.
    labels = pd.Series(labels, index=matrix.index)
    rows = []
    for g in sorted(labels.unique()):
        members = matrix.loc[labels == g]
        size = len(members)
        counts = members.sum(axis=0).sort_values(ascending=False)
        for code, count in counts.head(top_n).items():
            if count == 0:
                break
            rows.append({
                "Group": g,
                "Code": code,
                "Count": int(count),
                "Proportion": round(float(count) / size, 3),
                "Group Size": size,
            })
    return pd.DataFrame(rows)
