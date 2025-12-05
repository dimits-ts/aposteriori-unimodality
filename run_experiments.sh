#!/bin/bash

parallel --jobs 4 --delay 0.1 \
    ::: \
    'python -m src.variance_analysis --hundred-dataset-path=data/annotation.csv --dices-small-path=data/dices/350/diverse_safety_adversarial_dialog_350.csv --dices-large-path=data/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --cache-dir=cache' \
    'python -m src.explanation --graph-output-dir=graphs' \
    'python -m src.dices --dataset-small-path=data/dices/350/diverse_safety_adversarial_dialog_350.csv --dataset-large-path=data/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --latex-output-dir=manuscript/generated' \
    'python -m src.real_life_sap --dataset-path=data/sap.csv --output-dir=output --graph-output-dir=graphs' \
    'python -m src.real_life_kumar --dataset-path=data/kumar.json --output-dir=output --graph-output-dir=graphs' \
    'python -m src.synthetic_100 --dataset-path=data/annotation.csv --output-dir=output --graph-output-dir=graphs' 