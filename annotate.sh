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

# Datasets for the no-persona baseline (same as the main runs, so the two
# conditions can be compared directly).
no_persona_datasets=("${main_annotation_datasets[@]}")

# ============================================================
# Hardware configuration
# ============================================================
# Number of GPUs available: 1 or 2. With a single GPU, only models with
# at most SINGLE_GPU_MAX_PARAMS_B billion parameters are run.
NUM_GPUS=2
SINGLE_GPU_MAX_PARAMS_B=25

if [ "$NUM_GPUS" -ne 1 ] && [ "$NUM_GPUS" -ne 2 ]; then
  echo "NUM_GPUS must be 1 or 2 (got: ${NUM_GPUS})" >&2
  exit 1
fi

# ============================================================
# Model configuration
# ============================================================
declare -A model_ids=(
  [olmo32b]="unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
  [qwen32b]="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
  [olmo7b]="unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
  [qwen7b]="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  [llama8b]="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
  [mistral24b]="unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
  [mistral7b]="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  [gemma7b]="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
  [gemma31b]="unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
)

# Parameter count of each model, in billions (rounded up to an integer).
declare -A model_params_b=(
  [olmo32b]=32
  [qwen32b]=33
  [olmo7b]=7
  [qwen7b]=8
  [llama8b]=8
  [mistral24b]=24
  [mistral7b]=7
  [gemma7b]=6
  [gemma31b]=31
)

# Models for the main and ablation runs (all) and the adversarial runs (subset).
all_models=("olmo32b" "qwen32b" "olmo7b" "qwen7b" "llama8b" "mistral24b" "mistral7b" "gemma7b")
adv_models=("olmo32b" "qwen32b" "olmo7b" "qwen7b" "llama8b" "mistral24b" "mistral7b" "gemma7b")

# With a single GPU, drop models above the parameter limit.
# filter_models_for_gpus <models_array_name>
filter_models_for_gpus() {
  local -n models_ref="$1"
  [ "$NUM_GPUS" -eq 1 ] || return 0

  local kept=() model
  for model in "${models_ref[@]}"; do
    if [ -z "${model_params_b[$model]:-}" ]; then
      echo "No parameter count set for model '${model}' in model_params_b" >&2
      exit 1
    fi
    if [ "${model_params_b[$model]}" -le "$SINGLE_GPU_MAX_PARAMS_B" ]; then
      kept+=("$model")
    else
      echo "NUM_GPUS=1: skipping ${model} (${model_params_b[$model]}B > ${SINGLE_GPU_MAX_PARAMS_B}B)" >&2
    fi
  done
  models_ref=("${kept[@]}")
}

filter_models_for_gpus all_models
filter_models_for_gpus adv_models

# Anything above the number of annotators is wasted.
batch_size=20

# ============================================================
# Directory / run configuration
# ============================================================
instructions_dir="instructions/main"                 # <dir>/<dataset>/*
adv_instructions_dir="instructions/adversarial"      # <dir>/<dataset>/*
ablation_instructions_dir="instructions/ablation"    # <dir>/<dataset>/* (paraphrases)
no_persona_instructions_dir="instructions/main" # <dir>/<dataset>/*

output_dir="output/llm/annotations"
ablation_output_dir="output/llm/ablations"
ablation_repeat_output_dir="${ablation_output_dir}/repeat"
ablation_paraphrase_output_dir="${ablation_output_dir}/paraphrase"
# No-persona runs go next to the main runs; they are told apart by the
# "nopersona" prompt name in the file name (see run_annotation).
no_persona_output_dir="$output_dir"
no_persona_prompt_name="nopersona"

log_dir="logs"
log_file="${log_dir}/annotation.log"
ablation_log_file="${log_dir}/ablation.log"
no_persona_log_file="${log_dir}/no_persona.log"

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
#       no_persona (optional; non-empty = pass --no-persona)
run_annotation() {
  local dataset="$1"
  local instruction_path="$2"
  local model="$3"
  local out_dir="$4"
  local n_annotators="$5"
  local sample_fraction="$6"
  local suffix="$7"
  local target_log="$8"
  local no_persona="${9:-}"

  # No-persona runs are named after the condition rather than the prompt
  # file, so they do not clash with the persona runs of the same prompt.
  local instruction_name
  if [ -n "$no_persona" ]; then
    instruction_name="$no_persona_prompt_name"
  else
    instruction_name="$(basename "$instruction_path")"
    instruction_name="${instruction_name%.*}"
  fi

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
    --log-file                "$target_log"
  )
  [ -n "$sample_fraction" ] && cmd+=(--sample-fraction "$sample_fraction")
  [ -n "$no_persona" ] && cmd+=(--no-persona)

  # annotate.py writes its own log (incl. library warnings and tracebacks)
  # to $target_log; the progress bar stays on the terminal only.
  "${cmd[@]}"
  local status=$?

  if [ "$status" -ne 0 ]; then
    echo "FAILED (exit ${status}): ${label}" | tee -a "$target_log"
    return "$status"
  fi

  echo "Finished ${label}." | tee -a "$target_log"
}

# Run every instruction file in a directory for every model in a list.
# Args: dataset  instruction_dir  models_array_name  out_dir  num_annotators
#       sample_fraction  n_repeats (0 = run once, no suffix)  log
#       no_persona (optional; non-empty = pass --no-persona)
run_instruction_dir() {
  local dataset="$1"
  local instruction_dir="$2"
  local -n models_ref="$3"
  local out_dir="$4"
  local n_annotators="$5"
  local sample_fraction="$6"
  local n_repeats="$7"
  local target_log="$8"
  local no_persona="${9:-}"

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
            "$n_annotators" "$sample_fraction" "-run${run}" "$target_log" \
            "$no_persona"
        done
      else
        run_annotation "$dataset" "$instruction_path" "$model" "$out_dir" \
          "$n_annotators" "$sample_fraction" "" "$target_log" "$no_persona"
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

  # 5. No-persona baseline: same setup as the main runs, but no persona is
  #    appended to the prompt.
  if contains "$dataset" "${no_persona_datasets[@]}"; then
    banner "$no_persona_log_file" "STARTING NO-PERSONA ANNOTATIONS FOR DATASET: ${dataset}"
    run_instruction_dir "$dataset" "${no_persona_instructions_dir}/${dataset}" \
      all_models "$no_persona_output_dir" "$num_annotators" "" 0 \
      "$no_persona_log_file" 1
  fi
done

echo -e "\nAll annotation, ablation, adversarial, and no-persona runs completed."
echo "Main log:       $log_file"
echo "Ablation log:   $ablation_log_file"
echo "No-persona log: $no_persona_log_file"