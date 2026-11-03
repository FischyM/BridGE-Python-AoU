#!/bin/bash
set -e
set -u
set -o pipefail


# get the arguments from the command line
hh=$1                   # requested walltime in hours
mm=$2                   # requested walltime in minutes
cores=$3                # requested cpus cores
mem=$4                  # requested RAM in GB
node=$5                 # what node to run on: ex. ag2tb, agsmall

proj_dir=$6             # project directory that must be a subdirectory of the BridGE-Python-AoU directory
R=$7                    # number of random networks (real is 0)
pval=$8                 # p-value threshold for false discovery rate calculation

# move to slurm script directory
cd "/projects/standard/myersc/fisch872/bridge-projects/BridGE-Python-AoU/$proj_dir/slurm" || exit
echo


# create a file for the slurm script
file="4-fdr.txt"
if [ -f "$file" ]; then
    rm -f "$file"
fi
echo "$file"
touch "$file"


# write to the slurm script file
{
echo "#!/bin/bash -l"
echo
echo "#SBATCH --time=$hh:$mm:00"                # requested walltime in hours:minutes:seconds
echo "#SBATCH --nodes=1"                        # number of nodes to use for the job
echo "#SBATCH --ntasks=1"                       # number of tasks (jobs) to run on the allocated nodes
echo "#SBATCH --cpus-per-task=$cores"           # number of CPU cores per task (job)
echo "#SBATCH --mem=$mem"G                      # amount of RAM to allocate for the job
echo "#SBATCH --tmp=10G"                        # amount of local disk space to allocate for the job
echo "#SBATCH --mail-user=fisch872@umn.edu"     # email address to send notifications to
echo "#SBATCH --mail-type=FAIL,REQUEUE,END"     # types of email notifications to send
echo "#SBATCH --requeue"                        # allow the job to be requeued if it fails
echo "#SBATCH -A myersc"                        # account to charge the job to
echo
echo "source ~/.bashrc"
echo "source activate bridge3"
echo "cd /projects/standard/myersc/fisch872/bridge-projects/BridGE-Python-AoU"
echo "source setup.sh"
echo "python bridge.py --projectDir=$proj_dir --module=ComputeFDR --model=combined --pvalueCutoff=$pval --R=$R"
echo 
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
