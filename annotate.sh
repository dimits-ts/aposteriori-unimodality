#!/bin/bash
set -o pipefail

# ============================================================
# Dataset configuration
# ============================================================
datasets=("sap" "kumar" "dices-350" "dices-990")
dataset_paths=(
  "data/datasets/sap.csv"
  "data/datasets/kumar.json"
  "data/datasets/dices/350/diverse_safety_adversarial_dialog_350.csv"
  "data/datasets/dices/990/diverse_safety_adversarial_dialog_990.csv"
)
# Instructions subdirectory key for each dataset
instruction_keys=("sap" "kumar" "dices-350" "dices-990")

# ============================================================
# Model configuration
# ============================================================
# Models used for main + ablation runs (all 5)
all_models=(
  "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
  "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
  "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
all_pseudos=(
  "olmo32b"
  "qwen32b"
  "olmo7b"
  "qwen7b"
  "llama8b"
)

# Models used for adversarial runs (subset: 3 models)
adv_models=(
  "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
  "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
)
adv_pseudos=(
  "olmo32b"
  "qwen32b"
  "qwen7b"
)

# Batch size per model. Anything above the number of annotators is wasted.
declare -A batch_sizes=(
  [olmo32b]=20
  [qwen32b]=20
  [olmo7b]=20
  [qwen7b]=20
  [llama8b]=20
)

# Datasets that also get adversarial annotation (indices into the main arrays)
adv_dataset_indices=(0 1)   # sap, kumar

# ============================================================
# Directory configuration
# ============================================================
instructions_dir="instructions"
adv_instructions_dir="instructions/adversarial"

num_annotators=20

output_dir="output/llm/annotations"
log_dir="logs"
log_file="${log_dir}/annotation.log"

# Ablation settings
ablation_num_annotators=6
ablation_sample_fraction="0.1"
ablation_n_repeats=5
# Indexed like `datasets` (sap, kumar)
ablation_paraphrase_dirs=(
  "instructions/ablation/sap"
  "instructions/ablation/kumar"
  "instructions/ablation/dices-350"
  "instructions/ablation/dices-990"
)
ablation_output_dir="output/llm/ablations"
ablation_repeat_output_dir="${ablation_output_dir}/repeat"
ablation_paraphrase_output_dir="${ablation_output_dir}/paraphrase"
ablation_log_file="${log_dir}/ablation.log"

mkdir -p \
  "$output_dir" \
  "$log_dir" \
  "$ablation_repeat_output_dir" \
  "$ablation_paraphrase_output_dir"

# ============================================================
# Shared annotation runner
# ============================================================
# Args: dataset  dataset_path  instruction_path  model  pseudo
#       out_dir  num_annotators  sample_fraction  suffix  target_log
run_annotation() {
  local dataset="$1"
  local dataset_path="$2"
  local instruction_path="$3"
  local model="$4"
  local pseudo="$5"
  local out_dir="$6"
  local n_annotators="$7"
  local sample_fraction="$8"   # empty string => full dataset
  local suffix="$9"            # empty string => no suffix
  local target_log="${10}"

  local instruction_name
  instruction_name="$(basename "$instruction_path")"
  instruction_name="${instruction_name%.*}"

  local output_path="${out_dir}/${dataset}-${instruction_name}-${pseudo}${suffix}.csv"

  if [ -f "$output_path" ]; then
    echo "Skipping (already exists): ${output_path}" | tee -a "$target_log"
    return 0
  fi

  echo -e "\n=== Dataset: ${dataset} | Instruction: ${instruction_name}${suffix} x ${pseudo} (${model}) ===" >> "$target_log"

  local cmd=(python -m src.llm.annotate
    --dataset                 "$dataset"
    --dataset-path            "$dataset_path"
    --instruction-prompt-path "$instruction_path"
    --model-name              "$model"
    --output-path             "$output_path"
    --batch-size              "${batch_sizes[$pseudo]}"
    --num-annotators          "$n_annotators"
  )
  [ -n "$sample_fraction" ] && cmd+=(--sample-fraction "$sample_fraction")

  "${cmd[@]}" 2>&1 | tee -a "$target_log"
  local status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "FAILED (exit ${status}): ${dataset} - ${instruction_name}${suffix} x ${pseudo}" | tee -a "$target_log"
    return "$status"
  fi

  echo "Finished ${dataset} - ${instruction_name}${suffix} x ${pseudo}." | tee -a "$target_log"
}

# ============================================================
# Main loop over datasets
# ============================================================
for i in "${!datasets[@]}"; do
  current_dataset="${datasets[$i]}"
  current_dataset_path="${dataset_paths[$i]}"
  current_instructions_dir="${instructions_dir}/main/${instruction_keys[$i]}"
  current_paraphrase_dir="${ablation_paraphrase_dirs[$i]}"

  echo -e "\n\n=======================================================" >> "$log_file"
  echo "STARTING ANNOTATIONS FOR DATASET: ${current_dataset}"          >> "$log_file"
  echo "======================================================="        >> "$log_file"

  

  # ----------------------------------------------------------
  # 2. Repeat ablation: same prompt N times over a 10% sub-sample
  # ----------------------------------------------------------
  if [ -d "$current_instructions_dir" ]; then
    for instruction_path in "$current_instructions_dir"/*; do
      [ -f "$instruction_path" ] || continue
      for j in "${!all_models[@]}"; do
        for run in $(seq 0 $((ablation_n_repeats - 1))); do
          run_annotation \
            "$current_dataset"            \
            "$current_dataset_path"       \
            "$instruction_path"           \
            "${all_models[$j]}"           \
            "${all_pseudos[$j]}"          \
            "$ablation_repeat_output_dir" \
            "$ablation_num_annotators"    \
            "$ablation_sample_fraction"   \
            "-run${run}"                  \
            "$ablation_log_file"
        done
      done
    done
  else
    echo "Skipping repeat ablation for ${current_dataset}: no directory at ${current_instructions_dir}" | tee -a "$ablation_log_file"
  fi

  # ----------------------------------------------------------
  # 3. Paraphrase ablation: N similar prompts, each run once
  # ----------------------------------------------------------
  if [ -d "$current_paraphrase_dir" ]; then
    for instruction_path in "$current_paraphrase_dir"/*; do
      [ -f "$instruction_path" ] || continue
      for j in "${!all_models[@]}"; do
        run_annotation \
          "$current_dataset"                \
          "$current_dataset_path"           \
          "$instruction_path"               \
          "${all_models[$j]}"               \
          "${all_pseudos[$j]}"              \
          "$ablation_paraphrase_output_dir" \
          "$ablation_num_annotators"        \
          "$ablation_sample_fraction"       \
          ""                                \
          "$ablation_log_file"
      done
    done
  else
    echo "Skipping paraphrase ablation for ${current_dataset}: no directory at ${current_paraphrase_dir}" | tee -a "$ablation_log_file"
  fi

  
done

echo -e "\nAll annotation, ablation, and adversarial runs completed."
echo "Main log:     $log_file"
echo "Ablation log: $ablation_log_file"