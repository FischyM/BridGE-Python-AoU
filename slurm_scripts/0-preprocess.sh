#!/bin/bash
set -e
set -u
set -o pipefail


# get the arguments from the command line
hh=$1                   # requested walltime in hours
mm=$2                   # requested walltime in minutes
tasks=$3                # requested cpus cores
mem=$4                  # requested RAM in GB
node=$5                 # what node to run on: ex. ag2tb, agsmall

proj_dir=$6             # project directory that must be a subdirectory of the BridGE-Python-AoU directory
plink_name=$7           # name of the plink file to preprocess (without extensions), must be in the raw directory of the project directory
prj1000File=$8          # path to the prj1000 file, must be in the raw directory of the project directory
popIDFile=$9            # path to the population ID file, must be in the raw directory of the project directory
x1=${10}                # x1 coordinate for outlier removal
x2=${11}                # x2 coordinate for outlier removal
y1=${12}                # y1 coordinate for outlier removal
y2=${13}                # y2 coordinate for outlier removal


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
echo "#SBATCH --time=$hh:$mm:00"
echo "#SBATCH --ntasks=$tasks"                      # processor cores
echo "#SBATCH --mem=$mem"G                          # ram
echo "#SBATCH --tmp=10G"                            # local disk space
echo "#SBATCH --mail-type=FAIL,REQUEUE,END"         # FAIL or have option ALL
echo "#SBATCH --mail-user=fisch872@umn.edu"
echo "#SBATCH --requeue"
echo "#SBATCH -A myersc"
echo
echo "source ~/.bashrc"
echo "source activate bridge-aou"
echo "cd /projects/standard/myersc/fisch872/bridge-projects/BridGE-Python-AoU"
echo "source setup.sh"
echo "time ./scripts/check_population.sh $proj_dir/raw/$plink_name $proj_dir/raw/$prj1000File $proj_dir/raw/$popIDFile $proj_dir/preprocess/$plink_name.prj1000"
echo "time ./scripts/remove_outlier.sh $proj_dir/raw/$plink_name $proj_dir/preprocess/$plink_name.prj1000.eigenvec $proj_dir/preprocess/$plink_name.rmoutlier $x1 $x2 $y1 $y2"
echo "time ./scripts/preprocess.sh $proj_dir/preprocess/$plink_name.rmoutlier $proj_dir/raw/gwas_final"
echo 
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
