#!/bin/bash
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --time=20:00:00 # Runtime
#SBATCH --mem=200000 # Total memory pool for one or more cores (see also --mem-per-cpu)
#SBATCH -o slurm_results_%A.out # Standard out goes to this file
#SBATCH -e slurm_results_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL        # Send mail when process begins, fails, or finishes
#SBATCH --mail-user=yasuko_isoe@fas.harvard.edu

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=52
export MOVIE_PATH="/n/boslfs/LABS/engert_lab/Yasuko/medaka_movies"
export PYTHONUNBUFFERED=TRUE
#edit
module load Anaconda3/5.0.1-fasrc02
source activate py37

echo "Running Python on cluster...."
echo $*
hostname
date

srun python $*

echo "... Finished"
date