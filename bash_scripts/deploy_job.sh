#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --job-name=train_supervised_small
#SBATCH --partition=mundus 
#SBATCH --gres=gpu:a100-5:1
#SBATCH --cpus-per-task=16
#SBATCH --output=../logs/train_supervised_small.out
#SBATCH --error=../logs/err_train_supervised_small.out

# Check if arguments are provided
#export WANDB_API_KEY=


# Assign names to input arguments
year=$1
toy="false"

# Check if the optional argument is provided
# Parse command-line arguments
while getopts ":t" opt; do
  case ${opt} in
    t )
      toy=true
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

DATASET_PATH=../datasets/$year
# Print the named arguments
ml load cuda/11.0
ml load libcudnn
cd ..
nvidia-smi
echo "Starting training process"
python -m supervised.train --toy $toy
