#!/bin/bash
set -o pipefail

# ============================================================
# Dataset configuration
# ============================================================
# All datasets get the ablation runs (repeat + paraphrase).
datasets=("sap" "kumar" "dices-350" "dices-990" "popquorn")

declare -A dataset_paths=(
  [sap]="data/datasets/sap.csv"
  [kumar]="data/datasets/kumar.json"
  [dices-350]="data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv"
  [dices-990]="data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv"
  [popquorn]="data/datasets/popquorn_offensiveness.csv"
)

# Only these datasets get the main AND the adversarial annotation runs.
main_annotation_datasets=("sap" "kumar" "popquorn")

# ============================================================
# Model configuration
# ============================================================
declare -A model_ids=(
  [olmo32b]="unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
  [qwen32b]="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
  [olmo7b]="unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
  [qwen7b]="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  [llama8b]="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)

# Models for the main and ablation runs (all) and the adversarial runs (subset).
all_models=("olmo32b" "qwen32b" "olmo7b" "qwen7b" "llama8b")
adv_models=("olmo32b" "qwen32b" "qwen7b")

# Anything above the number of annotators is wasted.
batch_size=20

# ============================================================
# Directory / run configuration
# ============================================================
instructions_dir="instructions/main"                 # <dir>/<dataset>/*
adv_instructions_dir="instructions/adversarial"      # <dir>/<dataset>/*
ablation_instructions_dir="instructions/ablation"    # <dir>/<dataset>/* (paraphrases)

output_dir="output/llm/annotations"
ablation_output_dir="output/llm/ablations"
ablation_repeat_output_dir="${ablation_output_dir}/repeat"
ablation_paraphrase_output_dir="${ablation_output_dir}/paraphrase"

log_dir="logs"
log_file="${log_dir}/annotation.log"
ablation_log_file="${log_dir}/ablation.log"

num_annotators=20

# Ablation settings
ablation_num_annotators=6
ablation_sample_fraction="0.1"
ablation_n_repeats=5

mkdir -p \
  "$output_dir" \
  "$log_dir" \
  "$ablation_repeat_output_dir" \
  "$ablation_paraphrase_output_dir"

# ============================================================
# Helpers
# ============================================================
# contains <item> <list...>
contains() {
  local item="$1"
  shift
  local x
  for x in "$@"; do
    [ "$x" = "$item" ] && return 0
  done
  return 1
}

# banner <log> <text>
banner() {
  {
    echo -e "\n\n======================================================="
    echo "$2"
    echo "======================================================="
  } >> "$1"
}

# Run (or skip, if the output already exists) a single annotation job.
# Args: dataset  instruction_path  model  out_dir  num_annotators
#       sample_fraction (empty = full dataset)  suffix (may be empty)  log
run_annotation() {
  local dataset="$1"
  local instruction_path="$2"
  local model="$3"
  local out_dir="$4"
  local n_annotators="$5"
  local sample_fraction="$6"
  local suffix="$7"
  local target_log="$8"

  local instruction_name
  instruction_name="$(basename "$instruction_path")"
  instruction_name="${instruction_name%.*}"

  local label="${dataset} - ${instruction_name}${suffix} x ${model}"
  local output_path="${out_dir}/${dataset}-${instruction_name}-${model}${suffix}.csv"

  if [ -f "$output_path" ]; then
    echo "Skipping (already exists): ${output_path}" | tee -a "$target_log"
    return 0
  fi

  echo -e "\n=== ${label} (${model_ids[$model]}) ===" >> "$target_log"

  local cmd=(python -m src.llm.annotate
    --dataset                 "$dataset"
    --dataset-path            "${dataset_paths[$dataset]}"
    --instruction-prompt-path "$instruction_path"
    --model-name              "${model_ids[$model]}"
    --output-path             "$output_path"
    --batch-size              "$batch_size"
    --num-annotators          "$n_annotators"
  )
  [ -n "$sample_fraction" ] && cmd+=(--sample-fraction "$sample_fraction")

  "${cmd[@]}" 2>&1 | tee -a "$target_log"
  local status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "FAILED (exit ${status}): ${label}" | tee -a "$target_log"
    return "$status"
  fi

  echo "Finished ${label}." | tee -a "$target_log"
}

# Run every instruction file in a directory for every model in a list.
# Args: dataset  instruction_dir  models_array_name  out_dir  num_annotators
#       sample_fraction  n_repeats (0 = run once, no suffix)  log
run_instruction_dir() {
  local dataset="$1"
  local instruction_dir="$2"
  local -n models_ref="$3"
  local out_dir="$4"
  local n_annotators="$5"
  local sample_fraction="$6"
  local n_repeats="$7"
  local target_log="$8"

  if [ ! -d "$instruction_dir" ]; then
    echo "Skipping ${dataset}: no directory at ${instruction_dir}" | tee -a "$target_log"
    return 0
  fi

  local instruction_path model run
  for instruction_path in "$instruction_dir"/*; do
    [ -f "$instruction_path" ] || continue
    for model in "${models_ref[@]}"; do
      if [ "$n_repeats" -gt 0 ]; then
        for run in $(seq 0 $((n_repeats - 1))); do
          run_annotation "$dataset" "$instruction_path" "$model" "$out_dir" \
            "$n_annotators" "$sample_fraction" "-run${run}" "$target_log"
        done
      else
        run_annotation "$dataset" "$instruction_path" "$model" "$out_dir" \
          "$n_annotators" "$sample_fraction" "" "$target_log"
      fi
    done
  done
}

# ============================================================
# Main loop over datasets
# ============================================================
for dataset in "${datasets[@]}"; do
  banner "$log_file" "STARTING ANNOTATIONS FOR DATASET: ${dataset}"

  # 1 + 2. Main and adversarial annotations (same dataset list for both).
  if contains "$dataset" "${main_annotation_datasets[@]}"; then
    run_instruction_dir "$dataset" "${instructions_dir}/${dataset}" \
      all_models "$output_dir" "$num_annotators" "" 0 "$log_file"

    banner "$log_file" "STARTING ADVERSARIAL ANNOTATIONS FOR DATASET: ${dataset}"
    run_instruction_dir "$dataset" "${adv_instructions_dir}/${dataset}" \
      adv_models "$output_dir" "$num_annotators" "" 0 "$log_file"
  fi

  # 3. Repeat ablation: same (main) prompt N times over a sub-sample.
  run_instruction_dir "$dataset" "${instructions_dir}/${dataset}" \
    all_models "$ablation_repeat_output_dir" "$ablation_num_annotators" \
    "$ablation_sample_fraction" "$ablation_n_repeats" "$ablation_log_file"

  # 4. Paraphrase ablation: N similar prompts, each run once.
  run_instruction_dir "$dataset" "${ablation_instructions_dir}/${dataset}" \
    all_models "$ablation_paraphrase_output_dir" "$ablation_num_annotators" \
    "$ablation_sample_fraction" 0 "$ablation_log_file"
done

echo -e "\nAll annotation, ablation, and adversarial runs completed."
echo "Main log:     $log_file"
echo "Ablation log: $ablation_log_file"