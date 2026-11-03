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
plink_name=$7           # name of the plink file to preprocess (without extensions), must be in the raw directory of the project directory
output_name=$8          # name of the output file (without extensions)
prj1000File=$9          # name of the prj1000 file, must be in the raw directory of the project directory
popIDFile=${10}         # name of the population ID file, must be in the raw directory of the project directory
x1=${11}                # x1 coordinate for outlier removal
x2=${12}                # x2 coordinate for outlier removal
y1=${13}                # y1 coordinate for outlier removal
y2=${14}                # y2 coordinate for outlier removal


# move to slurm script directory
cd "/projects/standard/myersc/fisch872/bridge-projects/BridGE-Python-AoU/$proj_dir/slurm" || exit
echo


# create a file for the slurm script
file="0-preprocess.txt"
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
echo "./scripts/check_population.sh $proj_dir/raw/$plink_name $proj_dir/raw/$prj1000File $proj_dir/raw/$popIDFile $proj_dir/preprocess/$plink_name.prj1000"
echo "./scripts/remove_outlier.sh $proj_dir/raw/$plink_name $proj_dir/preprocess/$plink_name.prj1000.eigenvec $proj_dir/preprocess/rmoutlier $x1 $x2 $y1 $y2"
echo "./scripts/basic_QC.sh $proj_dir/preprocess/rmoutlier $proj_dir/preprocess/basicQC 0.05 0.01 0.000001 0.001 0.01"
echo "./scripts/prune_LD.sh $proj_dir/preprocess/basicQC $proj_dir/preprocess/pruneLD 50 5 0.1"
echo "./scripts/remove_related.sh $proj_dir/preprocess/pruneLD $proj_dir/preprocess/removedRelated 0.125"
echo "./scripts/match_CC.sh $proj_dir/preprocess/removedRelated $proj_dir/preprocess/$output_name 10 1"
echo "./scripts/required.sh $proj_dir/preprocess/$output_name $proj_dir/raw/$output_name $proj_dir/raw/glist-hg38-50000.bed 10"
echo
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
