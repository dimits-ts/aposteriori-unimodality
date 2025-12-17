python -m src.run_annotators \
    --output-path=data/toxicity .csv \
    --input-path=data/kumar.json \
    --hf-model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit" \
    --instructions-path=data/annotation/toxicity.txt

python -m src.run_annotators \
    --output-path=data/hate-speech.csv \
    --input-path=data/dices/350/diverse_safety_adversarial_dialog_350.csv \
    --hf-model="unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit" \
    --instructions-path=data/annotation/hate.txt