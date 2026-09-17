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
plink_name=$7           # name of the plink file to preprocess and must be in the raw subdirectory of $proj_dir


# move to slurm script directory
cd "/projects/standard/myersc/fisch872/BridGE-Python-AoU/$proj_dir/slurm" || exit
echo


# create a file for the slurm script
file="1-process_data.txt"
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
echo "cd /projects/standard/myersc/fisch872/BridGE-Python-AoU"
echo "source setup.sh"
echo "time python bridge.py --projectDir=$proj_dir --module=DataProcess --plinkFile=$plink_name --geneAnnotation=glist-hg38 --geneSets=c2.cp.v2026.1.Hs --simMeasure=either --jaccardCutoff=0.33 --overlapCutoff=0.5"
echo 
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
