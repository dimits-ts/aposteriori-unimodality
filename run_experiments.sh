#!/bin/bash

parallel --jobs 4 --delay 0.1 \
    ::: \
    'python -m src.variance_analysis --dataset-path=data/annotation.csv --graph-output-dir=graphs' \
    'python -m src.explanation --graph-output-dir=graphs' \
    'python -m src.dices --dataset-path-small=data/dices/350/diverse_safety_adversarial_dialog_350.csv --dataset-path-large=data/dices/990/diverse_safety_adversarial_dialog_990.csv --graph-output-dir=graphs --latex-output-dir=manuscript/generated' \
    'python -m src.real_life_sap --dataset-path=data/sap.csv --latex-output-dir=manuscript/generated --graph-output-dir=graphs' \
    'python -m src.real_life_kumar --dataset-path=data/kumar.json --latex-output-dir=manuscript/generated --graph-output-dir=graphs' \
    'python -m src.synthetic_100 --dataset-path=data/annotation.csv --latex-output-dir=manuscript/generated --graph-output-dir=graphs' 