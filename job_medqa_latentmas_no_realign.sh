#!/usr/bin/env bash
#SBATCH --job-name=medqa_latentmas_no_realign
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-4:00:00
#SBATCH -o logs/medqa_latentmas_no_realign_%j.out
#SBATCH -e logs/medqa_latentmas_no_realign_%j.err

set -e
cd /n/home03/jqin/LatentMAS-1
mkdir -p logs
module load python
source activate latentmas_new

python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task medqa --prompt hierarchical --max_samples 100 --latent_steps 20 --max_new_tokens 4096 --generate_bs 1 2>&1 | tee medqa_latentmas_100_no_realign.log
