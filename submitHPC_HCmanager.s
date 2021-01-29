#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ginkgo
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/slurm_%j.out

module purge

## executable
##SRCDIR=$HOME/ReclusterTreeAlgorithms/scripts
#
HCmanagerDIR=$SCRATCH/HCmanager
cd $HCmanagerDIR
mkdir -p logs


singularity exec --nv \
	    --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python $SCRATCH/HCmanager/src/HCmanager/run_evaluate.py --dataset_dir=$SCRATCH/ginkgo/data/invMassGinkgo/ --dataset=jets_6N_10trees_25tcut_${SLURM_ARRAY_TASK_ID}.pkl --NleavesMin=6 --output_dir=$SCRATCH/HCmanager/experiments/ginkgo/ --results_filename=outjets_6N_10trees_25tcut_${SLURM_ARRAY_TASK_ID}.pkl"



## to submit(for 3 jobs): sbatch --array 0-2 submitHPC_HCmanager.s





