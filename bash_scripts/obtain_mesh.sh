#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=obtain_mesh_2020
#SBATCH --partition=mundus 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=mesh_2020.out
#SBATCH --error=err_mesh_2020.txt

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "No year has been provided. Please provide at least one argument."
    exit 1
fi




# Assign names to input arguments
year=$1
DATASET_PATH=../datasets/$year
# Print the named arguments
ml load cuda/11.0
ml load colmap

echo "Obtaining the mesh for the data located in: $DATASET_PATH "
nvidia-smi
# colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sfm/ \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

 colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply
# echo "arg1: $arg1"
# echo "arg2: $arg2"
# echo "arg3: $arg3"