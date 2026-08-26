#!/bin/bash

set -uo pipefail

datasets=("dices-350" "dices-990" "sap" "kumar")
dataset_paths=(
    "data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv" 
    "data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv"
    "data/datasets/sap.csv"
    "data/datasets/kumar.json"
)
# --------------------------------

if [ ${#datasets[@]} -eq 0 ]; then
    echo "Error: No datasets defined. Please populate the 'datasets' and 'dataset_paths' arrays."
    exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <instructions-dir>"
    echo "Note: Datasets must be defined within the script."
    exit 1
fi

instructions_dir="instructions"

models=(
#"unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
#"unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
#"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
"unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
"unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
pseudos=(
#"olmo32b"
#"qwen32b"
#"llama70b"
"olmo7b"
"qwen7b"
"llama8b"
)

output_dir="output/annotations"
log_dir="logs"
log_file="${log_dir}/annotation.log"

mkdir -p "$output_dir" "$log_dir"

# Modified run_annotation to accept dataset and dataset_path
run_annotation() {
    local dataset="$1"
    local dataset_path="$2"
    local instruction_path="$3"
    local model="$4"
    local pseudo="$5"

    local instruction_name
    instruction_name="$(basename "$instruction_path")"
    instruction_name="${instruction_name%.*}"

    # Create a unique output path incorporating the dataset name
    local output_path="${output_dir}/${dataset}-${instruction_name}-${pseudo}.csv"

    if [ -f "$output_path" ]; then
        echo "Skipping (already exists): ${output_path}" | tee -a "$log_file"
        return
    fi

    echo -e "\n=== Dataset: ${dataset} | Instruction: ${instruction_name} x ${pseudo} (${model}) ===" >> "$log_file"

    python src/llm_annotate.py \
        --dataset "$dataset" \
        --dataset-path "$dataset_path" \
        --instruction-prompt-path "$instruction_path" \
        --model-name "$model" \
        --output-path "$output_path" \
        >> "$log_file" 2>&1

    echo "Finished ${dataset} - ${instruction_name} x ${pseudo}." | tee -a "$log_file"
}

# --- MAIN EXECUTION LOOP ---

# 1. Loop over all defined datasets
for i in "${!datasets[@]}"; do
    current_dataset="${datasets[$i]}"
    current_dataset_path="${dataset_paths[$i]}"

    echo -e "\n\n=======================================================" >> "$log_file"
    echo "STARTING ANNOTATIONS FOR DATASET: ${current_dataset}" >> "$log_file"
    echo "=======================================================" >> "$log_file"

    # 2. Loop over all instruction files in the specified directory
    for instruction_path in "$instructions_dir"/*; do
        if [ ! -f "$instruction_path" ]; then
            continue # Skip if it's not a regular file
        fi

        # 3. Loop over all models
        for j in "${!models[@]}"; do
            model="${models[$j]}"
            pseudo="${pseudos[$j]}"

            # Call the annotation function with all required parameters
            run_annotation \
                "$current_dataset" \
                "$current_dataset_path" \
                "$instruction_path" \
                "$model" \
                "$pseudo"
        done
    done
done

echo -e "\nAll annotation runs completed. Check $log_file for details."