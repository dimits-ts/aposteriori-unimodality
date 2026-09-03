#!/bin/bash

set -euo pipefail

run_and_log() {
    local job_name="$1"
    local cmd="$2"

    bash -c "$cmd" > "logs/${job_name}.log" 2>&1

    echo "Finished ${job_name}."
}

export -f run_and_log

mkdir -p logs

JOBS=(
'variance_analysis|python src/variance_analysis.py --dices-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dices-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --sap-path=data/datasets/sap.csv --kumar-path=data/datasets/kumar.json --graph-output-dir=graphs --cache-dir=cache --latex-output-dir=manuscript/generated'

'explanation|python src/explanation.py --graph-output-dir=graphs'

'dices|python src/dices.py --dataset-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dataset-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --output-dir=output/main --ablation-dir=ablation'

'sap|python src/sap.py --dataset-path=data/datasets/sap.csv --output-dir=output/main --graph-output-dir=graphs'

'metric_comparison | python src/metric_comparison.py --cache-path=cache/metric-comparison.csv --graph-output-path=graphs/metric_comparison.png'

'metric_comparison_simple | python src/metric_comparison.py --cache-path=cache/metric-comparison-simple.csv --graph-output-path=graphs/metric_comparison_simple.png --simple-simulation'

'kumar|python src/kumar.py --dataset-path=data/datasets/kumar.json --output-dir=output/main --graph-output-dir=graphs --ablation-dir=ablation'

'llm | python src/llm_analysis.py \
  --dices-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv \
  --dices-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv \
  --sap-path=data/datasets/sap.csv \
  --kumar-path=data/datasets/kumar.json \
  --annotations-dir=output/annotations \
  --paraphrase-dir=output/ablations/paraphrase \
  --graph-output-dir=graphs \
  --latex-output-dir=manuscript/generated \
  --apunim-output-dir=output/llm'
)

printf "%s\n" "${JOBS[@]}" |
parallel --colsep '\|' -j8 --delay 0.1 run_and_log {1} {2}

python src/export_results.py \
    --results-dir=output/main \
    --latex-output-dir=manuscript/generated \
    --graph-output-dir=graphs