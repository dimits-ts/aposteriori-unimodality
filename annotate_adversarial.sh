#!/bin/bash

set -uo pipefail

datasets=("sap" "kumar")
dataset_paths=(
"data/datasets/sap.csv"
"data/datasets/kumar.json"
)
# Instructions subdirectory to use for each dataset; the dices variants
# share the same annotation guidelines.
instruction_keys=("sap" "kumar")
# --------------------------------

instructions_dir="instructions/adversarial"

models=(
"unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
"unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
"unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
)
pseudos=(
"olmo32b"
"qwen32b"
"llama70b"
"qwen7b"
)

output_dir="output/annotations"
log_dir="logs"
log_file="${log_dir}/annotation.log"

mkdir -p "$output_dir" "$log_dir"

run_annotation() {
  local dataset="$1"
  local dataset_path="$2"
  local instruction_path="$3"
  local model="$4"
  local pseudo="$5"
  local out_dir="$6"
  local sample_fraction="$7"   # empty string => full dataset, normal run
  local suffix="$8"            # empty string => no suffix
  local target_log="$9"

  local instruction_name
  instruction_name="$(basename "$instruction_path")"
  instruction_name="${instruction_name%.*}"

  local output_path="${out_dir}/${dataset}-${instruction_name}-${pseudo}${suffix}.csv"

  if [ -f "$output_path" ]; then
    echo "Skipping (already exists): ${output_path}" | tee -a "$target_log"
    return
  fi

  echo -e "\n=== Dataset: ${dataset} | Instruction: ${instruction_name}${suffix} x ${pseudo} (${model}) ===" >> "$target_log"

  local cmd=(python src/llm_annotate.py
  --dataset "$dataset"
  --dataset-path "$dataset_path"
  --instruction-prompt-path "$instruction_path"
  --model-name "$model"
  --output-path "$output_path"
  )
  if [ -n "$sample_fraction" ]; then
    cmd+=(--sample-fraction "$sample_fraction")
  fi

  "${cmd[@]}" | tee -a "$target_log" 2>&1

  echo "Finished ${dataset} - ${instruction_name}${suffix} x ${pseudo}." | tee -a "$target_log"
}

# 1. Loop over all defined datasets
for i in "${!datasets[@]}"; do
  current_dataset="${datasets[$i]}"
  current_dataset_path="${dataset_paths[$i]}"
  current_instructions_dir="${instructions_dir}/${instruction_keys[$i]}"

  echo -e "\n\n=======================================================" >> "$log_file"
  echo "STARTING ANNOTATIONS FOR DATASET: ${current_dataset}" >> "$log_file"
  echo "=======================================================" >> "$log_file"

  if [ ! -d "$current_instructions_dir" ]; then
    echo "Skipping ${current_dataset}: no instructions directory at ${current_instructions_dir}" | tee -a "$log_file"
    continue
  fi

  # 2. Loop over the instruction files specific to this dataset
  for instruction_path in "$current_instructions_dir"/*; do
    if [ ! -f "$instruction_path" ]; then
      continue
    fi

    # 3. Loop over all models
    for j in "${!models[@]}"; do
      run_annotation \
        "$current_dataset" \
        "$current_dataset_path" \
        "$instruction_path" \
        "${models[$j]}" \
        "${pseudos[$j]}" \
        "$output_dir" \
        "" \
        "" \
        "$log_file"
    done
  done

done

echo -e "\nAll adversarial annotation runs completed. Check $log_file for details."
