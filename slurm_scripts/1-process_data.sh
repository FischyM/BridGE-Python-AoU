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

proj_dir=$6             # project directory that must be a subdirectory of the BridGE-Python directory
plink_name=$7           # name of the plink file to preprocess (without .bed/.bim/.fam extensions)


# move to slurm script directory
cd "/projects/standard/myersc/fisch872/BridGE-Python/$proj_dir/slurm" || exit
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
echo "#SBATCH --tmp=10G"                            # temp mem
echo "#SBATCH --mail-type=FAIL,REQUEUE,END"         # FAIL or have option ALL
echo "#SBATCH --mail-user=fisch872@umn.edu"
echo "#SBATCH --requeue"
echo "#SBATCH -A myersc"
echo
echo "source ~/.bashrc"
echo "conda activate BridGE-env"
echo "cd /projects/standard/myersc/fisch872/BridGE-Python"
echo "source setup.sh"
echo "time python bridge.py --projectDir=$proj_dir --job=DataProcess --plinkFile=$plink_name --geneAnnotation=glist-hg38 --genesets=c2.cp.v2026.1.Hs.filtered"
echo 
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
