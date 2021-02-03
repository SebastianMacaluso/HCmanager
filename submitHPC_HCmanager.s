#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
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

tcut=25

#-----------------------------------------------------------------------
#for algorithm in BeamSearchGreedy
#do
##SBATCH --time=1:00:00
##SBATCH --mem=32GB

##---------------
#for algorithm in ExactTrellis
#do
##SBATCH --time=1:00:00
##SBATCH --mem=32GB
#
#---------------
for algorithm in ExactAstar
do
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#
##---------------
#for algorithm in ApproxAstar
#do
##SBATCH --time=1:00:00
##SBATCH --mem=32GB

#---------------------------------------------
#  Nsamples=20
#  for minLeaves in 9

#  Nsamples=20
#  for minLeaves in 4 5 6 7 8 9 #sbatch --array 0-4 submitHPC_HCmanager.s

  #---------------
#  Nsamples=5
#  for minLeaves in 10 # sbatch --array 0-19 submitHPC_HCmanager.s

  #---------------
#  Nsamples=2
#  for minLeaves in 11 12 #sbatch --array 0-49 submitHPC_HCmanager.s

  #---------------
  Nsamples=1
  for minLeaves in 13 14  #sbatch --array 0-49 submitHPC_HCmanager.s

  do

    maxLeaves=$(( $minLeaves + 1 ))

    singularity exec --nv \
          --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
          /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
          bash -c "source /ext3/env.sh; python $SCRATCH/HCmanager/src/HCmanager/run_evaluate.py --dataset_dir=$SCRATCH/ginkgo/data/invMassGinkgo/ --dataset=jets_${minLeaves}N_${Nsamples}trees_${tcut}tcut_${SLURM_ARRAY_TASK_ID}.pkl --NleavesMin=$minLeaves --output_dir=$SCRATCH/HCmanager/experiments/ginkgo/ --results_filename=outjets_${minLeaves}N_${Nsamples}trees_${tcut}tcut_${SLURM_ARRAY_TASK_ID}.pkl --wandb_dir=$SCRATCH/HCmanager --max_leaves=20 --all_pairs_max_size=12 --algorithm=${algorithm}"

  done

done
## to submit(for 3 jobs): sbatch --array 0-2 submitHPC_HCmanager.s





