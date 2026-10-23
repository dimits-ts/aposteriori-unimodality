import argparse
from pathlib import Path

from ..lib import graphs
from ..lib.preprocessing import SapDataset
from ..lib import run_helper
from ..lib.util import skip_if_exists


def main(
    dataset_path: Path,
    output_dir: Path,
    graph_output_dir: Path,
    cache_dir: Path,
):
    graphs.graph_setup()
    ds = SapDataset(dataset_path=dataset_path)

    graphs.polarization_plot(ds=ds, output_path=graph_output_dir / "sap.png")

    res = run_helper.compute_inherent_polarization_exhaustive(dataset=ds)
    res.to_csv(
        output_dir / "sap-inherent.csv", header=True, index_label="comment"
    )
    run_helper.write_inherent_polarization_subsampled(
        ds, output_dir, name="sap"
    )

    ndfu_path = run_helper.human_ndfu_path(cache_dir, "sap")
    if not skip_if_exists(ndfu_path):
        run_helper.write_human_ndfu(ds, ndfu_path)

    res = run_helper.run_all_results(ds)
    res.to_csv(output_dir / "sap-results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the full dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the CSV result files.",
    )
    parser.add_argument(
        "--graph-output-dir",
        required=True,
        help="Directory for graphs.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for cached computations (e.g. human nDFU).",
    )
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir),
    )