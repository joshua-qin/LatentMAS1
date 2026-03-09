#!/usr/bin/env bash
#SBATCH --job-name=medqa_textmas_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH -o logs/medqa_textmas_full_%j.out
#SBATCH -e logs/medqa_textmas_full_%j.err
#SBATCH --mail-user=joshuaqin@college.harvard.edu
#SBATCH --mail-type=BEGIN,END

set -e
cd /n/home03/jqin/LatentMAS-1
mkdir -p logs
module load python
source activate latentmas_new

python run.py --method text_mas --model_name Qwen/Qwen3-4B --task medqa --prompt hierarchical --max_samples -1 --max_new_tokens 4096 --generate_bs 1 2>&1 | tee medqa_textmas_full.log
