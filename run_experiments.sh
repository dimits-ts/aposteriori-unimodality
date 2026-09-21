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
'variance_analysis|python -m src.synthetic.variance_analysis --dices-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dices-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --sap-path=data/datasets/sap.csv --kumar-path=data/datasets/kumar.json --graph-output-dir=graphs --cache-dir=cache --latex-output-dir=manuscript/generated'

'explanation|python -m src.synthetic.explanation --graph-output-dir=graphs'

'dices|python -m src.human.dices --dataset-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dataset-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --output-dir=output/human/main --ablation-dir=output/human/ablations'

'sap|python -m src.human.sap --dataset-path=data/datasets/sap.csv --output-dir=output/human/main --graph-output-dir=graphs'

'metric_comparison|python -m src.synthetic.metric_comparison --cache-path=cache/metric-comparison.csv --graph-output-path=graphs/metric_comparison.png'

'kumar|python -m src.human.kumar --dataset-path=data/datasets/kumar.json --output-dir=output/human/main --graph-output-dir=graphs --ablation-dir=output/human/ablations --latex-output-dir=manuscript/generated'

'llm|python -m src.llm.analysis --dices-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dices-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --sap-path=data/datasets/sap.csv --kumar-path=data/datasets/kumar.json --annotations-dir=output/llm/annotations --paraphrase-dir=output/llm/ablations/paraphrase --graph-output-dir=graphs --latex-output-dir=manuscript/generated --cache-dir=cache --exclude-models olmo7b llama8b'
)

printf "%s\n" "${JOBS[@]}" |
  parallel --colsep '\|' -j8 --delay 0.1 run_and_log "{1}" "{2}"

python -m src.human.export_results \
    --results-dir=output/human/main \
    --latex-output-dir=manuscript/generated \
    --graph-output-dir=graphs