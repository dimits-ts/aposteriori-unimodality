#!/bin/bash

parallel --jobs 4 --delay 0.1 \
    ::: \
    'python -m src.real_life_sap --dataset-path=data/sap.csv --latex-output-dir=manuscript/generated' \
    'python -m src.synthetic_100 --dataset-path=data/100_annotators.csv --latex-output-dir=manuscript/generated' \
    'python -m src.synthetic_vmd --dataset-path=data/vmd.csv --latex-output-dir=manuscript/generated' \
    'python -m src.real_life_kumar --dataset-path=data/kumar.json --latex-output-dir=manuscript/generated'