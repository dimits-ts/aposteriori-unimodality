#!/bin/bash

parallel --jobs 8 --delay 0.1 \
    ::: \
    'python src/variance_analysis.py --dices-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dices-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --sap-path=data/datasets/sap.csv --kumar-path=data/datasets/kumar.json --graph-output-dir=graphs --cache-dir=cache --latex-output-dir=manuscript/generated' \
    'python src/explanation.py --graph-output-dir=graphs' \
    'python src/dices.py --dataset-small-path=data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv --dataset-large-path=data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --output-dir=output' \
    'python src/sap.py --dataset-path=data/datasets/sap.csv --output-dir=output --graph-output-dir=graphs' \
    'python src/kumar.py --dataset-path=data/datasets/kumar.json --output-dir=output --graph-output-dir=graphs' \

python src/export_results.py \
    --results-dir=output --latex-output-dir=manuscript/generated/ \
    --graph-output-dir=graphs