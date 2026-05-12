# main.py
# entry point. two subcommands so the workflow stays clear:
#
#   1) parse    -> split free-text Q1 answers into bullet points
#                  (then the researcher manually codes them in Excel)
#   2) cluster  -> after manual coding is done, run the analysis
#
# argparse tutorial -> https://docs.python.org/3/howto/argparse.html

import argparse
import os
import pandas as pd

import parser as q1_parser   # avoid shadowing the stdlib "parser" module name
import data_loader
import encoding
import clustering
import plotting


def cmd_parse(args):
    print(f"loading {args.responses}")
    responses = q1_parser.load_responses(args.responses)
    print(f"  {len(responses)} participants with non-empty Q1")

    parsed = q1_parser.parse_participants(responses)
    print(f"  parsed into {len(parsed)} total bullet points")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # utf-8-sig so Excel opens unicode properly on Windows
    parsed.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  saved -> {args.output}")


def cmd_cluster(args):
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading taxonomy: {args.taxonomy}")
    taxonomy = data_loader.load_taxonomy(args.taxonomy)
    print(f"  {len(taxonomy)} grounds in taxonomy")

    print(f"loading coded data: {args.coded}")
    coded = data_loader.load_coded(args.coded)
    print(f"  {coded['Participant #'].nunique()} participants, {len(coded)} coded entries")

    print("building binary matrix")
    matrix = encoding.build_binary_matrix(coded, taxonomy)
    print(f"  shape: {matrix.shape[0]} participants x {matrix.shape[1]} grounds")
    matrix_path = os.path.join(out_dir, "binary_matrix.csv")
    matrix.to_csv(matrix_path, index_label="Participant #")
    print(f"  saved -> {matrix_path}")

    print("computing Jaccard distances")
    dist = clustering.compute_distance(matrix)

    print("running PCoA (3 dims)")
    coords, var_explained = clustering.run_pcoa(dist, n_dims=3)
    print(f"  variance explained: PCo1={var_explained[0]:.1f}%  "
          f"PCo2={var_explained[1]:.1f}%  PCo3={var_explained[2]:.1f}%")
    pcoa_df = pd.DataFrame(coords, index=matrix.index, columns=["PCo1", "PCo2", "PCo3"])
    pcoa_path = os.path.join(out_dir, "pcoa_coordinates.csv")
    pcoa_df.to_csv(pcoa_path, index_label="Participant #")
    print(f"  saved -> {pcoa_path}")

    print("running KMeans (silhouette picks k)")
    km_labels, best_k, ks, inertias, sils = clustering.run_kmeans(coords)
    print(f"  best k = {best_k}  (silhouette = {max(sils):.3f})")
    km_df = pd.DataFrame({"Participant #": matrix.index, "Cluster": km_labels})
    km_path = os.path.join(out_dir, "kmeans_assignments.csv")
    km_df.to_csv(km_path, index=False)
    print(f"  saved -> {km_path}")

    # let user override the k used for hierarchical, otherwise use what KMeans picked
    hk = args.k if args.k else best_k
    print(f"running hierarchical clustering (complete linkage, k={hk})")
    hier_labels, linkage_matrix = clustering.run_hierarchical(dist, k=hk)
    hier_df = pd.DataFrame({"Participant #": matrix.index, "Cluster": hier_labels})
    hier_path = os.path.join(out_dir, "hierarchical_assignments.csv")
    hier_df.to_csv(hier_path, index=False)
    print(f"  saved -> {hier_path}")

    print("making plots")
    plotting.plot_pcoa_2d(coords, km_labels, var_explained,
                          os.path.join(out_dir, "pcoa_2d.png"))
    plotting.plot_pcoa_3d(coords, km_labels, var_explained,
                          os.path.join(out_dir, "pcoa_3d.png"))
    plotting.plot_elbow_silhouette(ks, inertias, sils, best_k,
                                   os.path.join(out_dir, "elbow_silhouette.png"))
    plotting.plot_dendrogram(linkage_matrix,
                             os.path.join(out_dir, "dendrogram.png"))
    print("done.")


def main():
    p = argparse.ArgumentParser(description="HCI objection clustering pipeline (simple version)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="split Q1 free-text into bullet points")
    p_parse.add_argument("--responses", required=True,
                         help="path to the Screening Survey Participant Responses xlsx")
    p_parse.add_argument("--output", default="output/parsed_points.csv",
                         help="where to save parsed_points.csv")
    p_parse.set_defaults(func=cmd_parse)

    p_cl = sub.add_parser("cluster", help="run binary encoding + PCoA + clustering")
    p_cl.add_argument("--coded", required=True, help="path to the completed coded xlsx")
    p_cl.add_argument("--taxonomy", required=True, help="path to the taxonomy xlsx")
    p_cl.add_argument("--output", default="output",
                      help="directory to write CSVs/PNGs into")
    p_cl.add_argument("--k", type=int, default=None,
                      help="override k for hierarchical clustering (defaults to KMeans best k)")
    p_cl.set_defaults(func=cmd_cluster)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
