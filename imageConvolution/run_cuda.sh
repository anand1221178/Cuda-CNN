#!/bin/bash

#-----------------------------------------------------------------------
# SLURM Job Script for CUDA Image Convolution Assignment
#-----------------------------------------------------------------------

# --- Resource Requests ---
#SBATCH --job-name=CUDA_Conv_Job
#SBATCH --partition=biggpu           # Use GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # CPU threads for preprocessing or helpers

#SBATCH --time=24:00:00

# --- Output Files ---
#SBATCH --output=cuda_conv_%j.out    # Standard output
#SBATCH --error=cuda_conv_%j.err     # Standard error

#-----------------------------------------------------------------------
# Job Execution Steps
#-----------------------------------------------------------------------

echo "========================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Node: $SLURM_JOB_NODELIST"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "========================================================"

echo "Loading required modules..."
module purge
module load gcc
module load cuda

module list

# Set working directory
WORK_DIR="/home-mscluster/panand/imageConvolution"
cd $WORK_DIR || { echo "Error changing directory to $WORK_DIR"; exit 1; }

echo "Current working directory: $(pwd)"

# Compile the CUDA program
echo "Compiling CUDA convolution program..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Make failed!"
    exit 1
fi
echo "Compilation successful."

# Run the executable
echo "Running the CUDA convolution executable..."
./imageConvolution

if [ $? -ne 0 ]; then
    echo "Executable failed!"
    exit 1
fi

echo "Execution completed."

#-----------------------------------------------------------------------
echo "========================================================"
echo "Job finished at $(date)"
echo "========================================================"
