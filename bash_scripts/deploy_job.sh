#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=train_supervised_small2
#SBATCH --partition=mundus 
#SBATCH --gres=gpu:a100-5:1
#SBATCH --cpus-per-task=16
#SBATCH --output=../logs/train_supervised_small2.out
#SBATCH --error=../logs/err_train_supervised_small2.out

# Check if arguments are provided
#export WANDB_API_KEY=


# Assign names to input arguments
toy=${1:-"false"}

# # Check if the optional argument is provided
# # Parse command-line arguments
# while getopts ":t" opt; do
#   case ${opt} in
#     t )
#       toy=true
#       ;;
#     \? )
#       echo "Invalid option: -$OPTARG" >&2
#       exit 1
#       ;;
#   esac
# done
# shift $((OPTIND -1))

# Print the named arguments
ml load cuda/11.0
ml load libcudnn
cd ..
nvidia-smi
echo "Starting training process"
echo "Toy flag set as $toy "



python -m supervised.train --toy $toy
