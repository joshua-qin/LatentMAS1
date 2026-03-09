#!/usr/bin/env bash
#SBATCH --job-name=medqa_ablate_corrupt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:15:00
#SBATCH -o logs/medqa_hier_ablation_corrupt_10_%j.out
#SBATCH -e logs/medqa_hier_ablation_corrupt_10_%j.err

set -e
cd /n/home03/jqin/LatentMAS-1
mkdir -p logs
module load python
source activate latentmas_new

# Ablation: hierarchical_fixed + no_rope (normally good), but SHUFFLE fused cache so judger sees random order.
# Expect bad accuracy. 25 samples only.
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task medqa --prompt hierarchical \
  --hierarchical_fixed --no_hierarchical_fixed_rope --hierarchical_fixed_corrupt_cache \
  --max_samples 25 --latent_steps 20 --latent_space_realign --max_new_tokens 4096 --generate_bs 1 --seed 42 \
  2>&1 | tee medqa_hier_ablation_corrupt_25.log
