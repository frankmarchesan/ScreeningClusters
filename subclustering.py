# subclustering.py
# post-clustering analysis: sub-cluster within parent clusters, and
# summarise the top taxonomy codes per (sub)cluster so patterns are readable.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans_sub(coords, parent_labels, sub_k=None, random_state=42, sub_k_max=6):
    # re-run KMeans within each parent cluster on the same PCoA coords.
    # sub_k=None  -> each parent picks its OWN best sub-k by silhouette (more optimal:
    #                tight clusters split less, messy ones split more, tiny ones not at all).
    # sub_k=int   -> force that many sub-clusters in every parent (old behaviour).
    # returns (sub, chosen): sub = 1D array of sub-cluster labels (1-indexed *per parent*),
    #         chosen = {parent_label: number_of_subclusters_used}.
    # TODO: fill body
    raise NotImplementedError


def top_codes(matrix, labels, top_n=3):
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
