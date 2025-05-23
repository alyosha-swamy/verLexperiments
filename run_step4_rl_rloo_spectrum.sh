#!/usr/bin/env bash
set -x # Added from Qwen example
set -euo pipefail

# Ensure the project directory (containing setup.py) is in PYTHONPATH
export PYTHONPATH="/home/ubuntu/verl:${PYTHONPATH:-}"

export CUDA_LAUNCH_BLOCKING=1

# ── User-tunable parameters ────────────────────────────────────────────────────
# Model from Step 3 (Samsungstep3)
BASE_MODEL_PATH="/home/ubuntu/Samsungstep3"
# Output directory for this Step 4 model
SAVE_DIR="./sky_t1_7B_step4_rloo_spectrum" # Changed save directory
# Eurus-2-RL-Data path.
EURUS_TRAIN_FILE="/home/ubuntu/verLexperiments/data/train_rloo_subsets_15k.parquet"
EURUS_VAL_FILE="/home/ubuntu/verLexperiments/data/validation_rloo_subsets.parquet"

# Spectrum Configuration
USE_SPECTRUM_FREEZING=True
export SPECTRUM_YAML_PATH="/home/ubuntu/verLexperiments/snr_results_Qwen-Qwen2.5-Math-7B_unfrozenparameters_50percent.yaml"

NPROC_PER_NODE=8 # Adapted from Qwen example (and user request)
# TOTAL_TRAINING_STEPS and other batch size variables are removed as values are now hardcoded from qwen2 example.

echo "🚀 Starting Step 4: RL Again (RLOO) with Spectrum Parameter Freezing (top-level sitecustomize) - Adapted from Qwen2-7B example"
echo " GPUs (trainer.n_gpus_per_node): $NPROC_PER_NODE"
echo " Base Model (actor_rollout_ref.model.path): $BASE_MODEL_PATH"
echo " Training Data (data.train_files): $EURUS_TRAIN_FILE"
echo " Validation Data (data.val_files): $EURUS_VAL_FILE"
echo " Output Directory (trainer.default_local_dir): $SAVE_DIR"
echo " Global Batch Size (data.train_batch_size): 512"
echo " PPO Mini Batch Size (actor_rollout_ref.actor.ppo_mini_batch_size): 128"
echo " Actor PPO Micro Batch Size per GPU (actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu): 40"
echo " Rollout Log Prob Micro Batch Size per GPU (actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu): 80"
echo " Reference Log Prob Micro Batch Size per GPU (actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu): 80"
echo " Max Prompt Length (data.max_prompt_length): 512"
echo " Max Response Length (data.max_response_length): 1024"
echo " Tensor Model Parallel Size (actor_rollout_ref.rollout.tensor_model_parallel_size): 2"
echo " GPU Memory Utilization (actor_rollout_ref.rollout.gpu_memory_utilization): 0.9"
echo " Rollouts (actor_rollout_ref.rollout.n): 5"
echo " Total Epochs (trainer.total_epochs): 15"
echo " Test Frequency (trainer.test_freq): 5"
echo " Project Name (trainer.project_name): verl_rloo_spectrum_patch_example_gsm8k" # Changed project name
echo " Experiment Name (trainer.experiment_name): qwen2_7b_spectrum_patch_function_rm" # Changed experiment name
echo " Using Spectrum Freezing: $USE_SPECTRUM_FREEZING"
echo " Spectrum YAML Path (env): $SPECTRUM_YAML_PATH"
echo "⚠️ If using subsets 'numina_amc_aime' and 'numina_olympiads', ensure the data loading handles this."
echo "   The current script uses the filtered files: $EURUS_TRAIN_FILE and $EURUS_VAL_FILE."

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d  "$BASE_MODEL_PATH" ]] || { echo "❌ Error: Base model directory not found: $BASE_MODEL_PATH"; exit 1; }
[[ -f  "$EURUS_TRAIN_FILE" ]] || { echo "❌ Error: Eurus train file not found: $EURUS_TRAIN_FILE"; exit 1; }
[[ -f  "$EURUS_VAL_FILE" ]] || { echo "❌ Error: Eurus validation file not found: $EURUS_VAL_FILE"; exit 1; }
if [ "$USE_SPECTRUM_FREEZING" = "True" ] && [ ! -f "$SPECTRUM_YAML_PATH" ]; then
    echo "❌ Error: Spectrum YAML file not found: $SPECTRUM_YAML_PATH"
    exit 1
fi

mkdir -p "$SAVE_DIR"

# ── Install Spectrum Patcher and Set Environment Variable ─────────────────────
echo "🔧 Ensuring verLexperiments repo is editable and SPECTRUM_YAML_PATH is set..."
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd)
/home/ubuntu/.venv/bin/python3 -m uv pip install ray[default] # Ensure ray is installed
/home/ubuntu/.venv/bin/python3 -m pip install -e "$SCRIPT_DIR" # Install the repo root in editable mode
export SPECTRUM_YAML_PATH # Export for sitecustomize.py to pick up
echo "✅ verLexperiments repo installed in editable mode and SPECTRUM_YAML_PATH exported."

# ── RLOO run with Spectrum (based on verl/examples/rloo_trainer/run_qwen2-7b.sh) ───────────
# Note: Parameters like learning rate, max_prompt/response_length, etc., are taken from the example.
# Adjust them as needed for your specific model and data.

# Prepare Spectrum flags for Hydra config (even if patch script uses env var, underlying code might check Hydra)
SPECTRUM_HYDRA_ARGS=""
if [ "$USE_SPECTRUM_FREEZING" = "True" ]; then
    SPECTRUM_HYDRA_ARGS="+data.use_spectrum_freezing=$USE_SPECTRUM_FREEZING +data.spectrum_yaml_path=\"$SPECTRUM_YAML_PATH\""
fi

echo "🚀 Launching RLOO training with Spectrum (relying on top-level sitecustomize.py for patching)..."
/home/ubuntu/.venv/bin/python3 -m verl.trainer.main_ppo \
    $SPECTRUM_HYDRA_ARGS \
    algorithm.adv_estimator=rloo \
    data.train_files="$EURUS_TRAIN_FILE" \
    data.val_files="$EURUS_VAL_FILE" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$BASE_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype="bfloat16" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.project_name='verl_rloo_spectrum_patch_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_spectrum_patch_function_rm' \
    trainer.logger=[console,wandb] \
    trainer.n_gpus_per_node="$NPROC_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15

echo "✅ RLOO Spectrum training script launched. Outputs will be in $SAVE_DIR" 