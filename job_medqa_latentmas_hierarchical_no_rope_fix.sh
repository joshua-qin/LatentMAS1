#!/usr/bin/env bash
#SBATCH --job-name=medqa_latent_hier_no_rope
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-4:00:00
#SBATCH -o logs/medqa_latentmas_hierarchical_no_rope_fix_%j.out
#SBATCH -e logs/medqa_latentmas_hierarchical_no_rope_fix_%j.err
#SBATCH --mail-user=joshuaqin@college.harvard.edu
#SBATCH --mail-type=BEGIN,END

set -e
cd /n/home03/jqin/LatentMAS-1
mkdir -p logs
module load python
source activate latentmas_new

# Same as hierarchical_fixed but with ROPE fix disabled (no unrotate->fuse->rotate)
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task medqa --prompt hierarchical --max_samples 100 --latent_steps 20 --latent_space_realign --hierarchical_fixed --no_hierarchical_fixed_rope --max_new_tokens 4096 --generate_bs 1 2>&1 | tee medqa_latentmas_100_hierarchical_no_rope_fix.log
