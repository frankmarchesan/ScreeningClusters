# data_loader.py
# small helpers for loading the two xlsx inputs.
# pandas docs for read_excel -> https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

import pandas as pd

def load_taxonomy(path):
    # taxonomy is a single-column list of Code values.
    df = pd.read_excel(path)
    df = df.iloc[:, :1]
    df.columns = ["Code"]
    df["Code"] = (
        df["Code"].astype(str)
        .str.replace(r'\s*\n\s*', ' ', regex=True)
        .str.strip()
    )
    df = df[df["Code"].ne("") & df["Code"].ne("nan")]
    df = df.drop_duplicates()
    return df


def load_coded(path):
    # the coded workbook has a sheet called "Coding" with this shape:
    #   Participant # | Point Index | Point | Code L1 | Code L2 | Code L3
    # we want it in long form: one row per (Participant #, Code).
    df = pd.read_excel(path, sheet_name="Coding")

    rows = []
    for _, row in df.iterrows():
        pid = row["Participant #"]
        point = row["Point"]
        for layer in range(1, 4): # L1, L2, L3
            ccol = f"Code L{layer}"
            if ccol not in df.columns:
                break
            code = row.get(ccol)
            if pd.notna(code) and str(code).strip():
                rows.append({
                    "Participant #": pid,
                    "Point": point,
                    "Code": str(code).strip(),
                })

    out = pd.DataFrame(rows)
    # same (Participant, Code) pair can show up twice across layers, drop dupes
    out = out.drop_duplicates(subset=["Participant #", "Code"])
    return out
