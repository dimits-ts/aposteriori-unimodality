import argparse
from pathlib import Path

from ..lib import graphs
from ..lib import run_helper
from ..lib.preprocessing import PopquornDataset
from ..lib.util import skip_if_exists


def main(
    dataset_path: Path,
    output_dir: Path,
    graph_output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    graphs.graph_setup()

    ds = PopquornDataset(dataset_path=dataset_path)
    name = ds.get_name().lower()

    graph_path = graph_output_dir / f"{name}.png"
    if not skip_if_exists(graph_path):
        graphs.polarization_plot(ds=ds, output_path=graph_path)

    inherent_path = output_dir / f"{name}-inherent.csv"
    if not skip_if_exists(inherent_path):
        res = run_helper.compute_inherent_polarization_exhaustive(dataset=ds)
        res.to_csv(inherent_path, header=True, index_label="comment")

    results_path = output_dir / f"{name}-results.csv"
    if not skip_if_exists(results_path):
        res = run_helper.run_all_results(ds)
        res.to_csv(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the POPQUORN offensiveness and politeness annotations."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the offensiveness dataset.",
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
    args = parser.parse_args()
    main(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
