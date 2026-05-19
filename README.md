### Requirements

Install dependencies for this to run:

```
pip install -r requirements.txt
```


### Commands to run for this

Parse open text objections which are grouped together into bullet points:

```
python main.py parse --responses "responses.xlsx" --output parsed_points.csv
```

Run clustering analysus on the coded data:

```
python main.py cluster --coded "pathtocoded.xlsx" --taxonomy "pathtotaxonomy.xlsx" --output output --k N --pcoa-k N
```

NOTE: `--k N` argument is optional

### Specifying K for hierarchical

`--k N` is optional. If you pass it, hierarchical clustering will use that exact N clusters. If you leave it out, script picks K automatically using the silhouette score from KMeans, and hierarchical clustering reuses same K.

### Specifying K for PCoA

`--pcoa-k N`

Same function as `--k N`, but for PCoA.


### Output

Cluster command writes these files into the output folder:

* `binary_matrix.csv`: participants by codes table with 0 or 1.
* `pcoa_coordinates.csv`: 3D PCoA coordinates for each participant.
* `kmeans_assignments.csv`: Participant # and KMeans cluster, sorted by cluster.
* `hierarchical_assignments.csv`: Participant # and hierarchical cluster (complete linkage), sorted by cluster.
* `pcoa_2d.png`, `pcoa_3d.png`, `elbow_silhouette.png`, `dendrogram.png`: plots.
