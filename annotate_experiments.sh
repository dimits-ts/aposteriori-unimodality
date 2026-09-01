#!/bin/bash

set -uo pipefail

datasets=("dices-350" "dices-990" "sap" "kumar")
dataset_paths=(
"data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv"
"data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv"
"data/datasets/sap.csv"
"data/datasets/kumar.json"
)
# Instructions subdirectory to use for each dataset; the dices variants
# share the same annotation guidelines.
instruction_keys=("dices" "dices" "sap" "kumar")
# --------------------------------

instructions_dir="instructions"

models=(
"unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
"unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
"unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
"unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
pseudos=(
"olmo32b"
"qwen32b"
"llama70b"
"olmo7b"
"qwen7b"
"llama8b"
)

output_dir="output/annotations"
log_dir="logs"
log_file="${log_dir}/annotation.log"

mkdir -p "$output_dir" "$log_dir"

# --------------------------------
# Sensitivity ablation settings
# --------------------------------
# Ablations run over a 10% sub-sample of each dataset instead of the full
# SAMPLES_PER_DATASET counts, to keep runtime manageable.
ablation_sample_fraction="0.1"
# Number of repeat runs for the "same prompt, N times" ablation.
ablation_n_repeats=5

ablation_paraphrase_dirs=(
"instructions/ablation/dices"
"instructions/ablation/dices"
"instructions/ablation/sap"
"instructions/ablation/kumar"
)

ablation_output_dir="output/ablations"
ablation_repeat_output_dir="${ablation_output_dir}/repeat"
ablation_paraphrase_output_dir="${ablation_output_dir}/paraphrase"
ablation_log_file="${log_dir}/ablation.log"

mkdir -p "$ablation_repeat_output_dir" "$ablation_paraphrase_output_dir"

# Shared runner for a single annotation call. Used for normal runs and for
# both ablation types -- they differ only in output dir, whether a sample
# fraction is passed, an optional output-filename suffix, and which log
# file gets the output.
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
  current_instructions_dir="${instructions_dir}/main/${instruction_keys[$i]}"
  # repeat the prompts of the main annotation
  current_repeat_dir="${current_instructions_dir}"
  current_paraphrase_dir="${ablation_paraphrase_dirs[$i]}"

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

  # 4. Repeat ablation: same prompt, run N times over a 10% sub-sample.
  if [ -d "$current_repeat_dir" ]; then
    for instruction_path in "$current_repeat_dir"/*; do
      [ -f "$instruction_path" ] || continue
      for j in "${!models[@]}"; do
        for run in $(seq 0 $((ablation_n_repeats - 1))); do
          run_annotation \
            "$current_dataset" \
            "$current_dataset_path" \
            "$instruction_path" \
            "${models[$j]}" \
            "${pseudos[$j]}" \
            "$ablation_repeat_output_dir" \
            "$ablation_sample_fraction" \
            "-run${run}" \
            "$ablation_log_file"
        done
      done
    done
  else
    echo "Skipping repeat ablation for ${current_dataset}: no directory at ${current_repeat_dir}" | tee -a "$ablation_log_file"
  fi

  # 5. Paraphrase ablation: N similar prompts, each run once over the same
  # 10% sub-sample.
  if [ -d "$current_paraphrase_dir" ]; then
    for instruction_path in "$current_paraphrase_dir"/*; do
      [ -f "$instruction_path" ] || continue
      for j in "${!models[@]}"; do
        run_annotation \
          "$current_dataset" \
          "$current_dataset_path" \
          "$instruction_path" \
          "${models[$j]}" \
          "${pseudos[$j]}" \
          "$ablation_paraphrase_output_dir" \
          "$ablation_sample_fraction" \
          "" \
          "$ablation_log_file"
      done
    done
  else
    echo "Skipping paraphrase ablation for ${current_dataset}: no directory at ${current_paraphrase_dir}" | tee -a "$ablation_log_file"
  fi

done

echo -e "\nAll annotation and ablation runs completed. Check $log_file and $ablation_log_file for details."
