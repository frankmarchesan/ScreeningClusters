# encoding.py
# turns the long-form coded dataframe into a participants x codes binary matrix.
# this is basically one-hot encoding but on a category that can appear multiple times per row.
#
# one-hot in pandas (good intro) -> https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
# why binary matrices for clustering text-coded data ->
#   https://en.wikipedia.org/wiki/Jaccard_index  (the distance we use later on this matrix)

import pandas as pd


def build_binary_matrix(coded_df, taxonomy_df):
    # use the taxonomy order for columns so the output is reproducible
    # (otherwise pandas will order them by first-seen which changes per run)
    code_cols = taxonomy_df["Code"].tolist()

    # add anything that got coded but isn't in the taxonomy (e.g. "Ambiguous")
    # - this reoccured in the old pipeline so handling it here on purpose
    for c in coded_df["Code"].dropna().unique():
        if c and c not in code_cols:
            code_cols.append(c)

    # one row per participant
    participants = sorted(coded_df["Participant #"].dropna().unique())

    # build an empty matrix of zeros and fill in 1s
    # could probably do this with crosstab but the explicit loop is easier to read
    # crosstab docs (for future me) -> https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
    matrix = pd.DataFrame(0, index=participants, columns=code_cols)

    for pid, sub in coded_df.groupby("Participant #"):
        codes = set(sub["Code"].dropna().tolist())
        for c in codes:
            if c in matrix.columns:
                matrix.at[pid, c] = 1

    # drop any column that's always zero (no one ever picked it).
    # otherwise Jaccard distance can divide by zero / get weird.
    matrix = matrix.loc[:, matrix.sum() > 0]

    return matrix
