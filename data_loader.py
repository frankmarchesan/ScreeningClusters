# data_loader.py
# small helpers for loading the two xlsx inputs.
# pandas docs for read_excel -> https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

import pandas as pd

# how many Domain/Ground "layers" we allow per coded point.
# the manual coding sheet uses Domain L1..L5, Objection Ground L1..L5.
MAX_LAYERS = 5


def load_taxonomy(path):
    # only the first two columns matter (Domain, Objection Ground).
    # the Domain column has merged-looking cells -> forward fill them.
    # ffill cheat sheet -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html
    df = pd.read_excel(path)
    df = df.iloc[:, :2]
    df.columns = ["Domain", "Objection Ground"]
    df["Domain"] = df["Domain"].ffill()
    # tidy up the domain names (sometimes they have linebreaks inside)
    df["Domain"] = df["Domain"].astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()
    df = df.dropna(subset=["Objection Ground"])
    return df


def load_coded(path):
    # the coded workbook has a sheet called "Coding" with this shape:
    #   Participant # | Point | Domain L1 | Objection Ground L1 | Domain L2 | ...
    # we want it in long form: one row per (Participant #, Domain, Ground).
    # this is basically a wide -> long melt by hand because the column pairs
    # don't fit the pd.melt pattern cleanly.
    # melt docs (for reference) -> https://pandas.pydata.org/docs/reference/api/pandas.melt.html
    df = pd.read_excel(path, sheet_name="Coding")

    rows = []
    for _, row in df.iterrows():
        pid = row["Participant #"]
        point = row["Point"]
        for layer in range(1, MAX_LAYERS + 1):
            dcol = f"Domain L{layer}"
            gcol = f"Objection Ground L{layer}"
            if dcol not in df.columns:
                break
            domain = row.get(dcol)
            ground = row.get(gcol)
            # skip blank cells
            if pd.notna(domain) and str(domain).strip():
                rows.append({
                    "Participant #": pid,
                    "Point": point,
                    "Domain": str(domain).strip(),
                    "Objection Ground": str(ground).strip() if pd.notna(ground) else "",
                })

    out = pd.DataFrame(rows)
    # same (Participant, Domain, Ground) triple can show up twice across layers - drop dupes
    out = out.drop_duplicates(subset=["Participant #", "Domain", "Objection Ground"])
    return out
