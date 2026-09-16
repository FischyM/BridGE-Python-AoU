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
x1=$8                   # x1 coordinate for outlier removal
x2=$9                   # x2 coordinate for outlier removal
y1=${10}                # y1 coordinate for outlier removal
y2=${11}                # y2 coordinate for outlier removal
prj1000File="/projects/standard/myersc/shared/BridGE_files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid"
popIDFile="/projects/standard/myersc/shared/BridGE_files/allpopid.txt"

# move to slurm script directory
cd "/projects/standard/myersc/fisch872/BridGE-Python/$proj_dir/slurm" || exit
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
echo "time ./scripts/data_checkpopulation.sh --plinkFile=$proj_dir/raw/$plink_name --prj1000File=$prj1000File --popIDFile=$popIDFile"
echo "time ./scripts/data_removeoutlier.sh --plinkFile=$proj_dir/raw/$plink_name --mdsFile=$proj_dir/intermediate/$plink_name.prj1000.mds --x1=$x1 --x2=$x2 --y1=$y1 --y2=$y2"
echo "time ./scripts/preprocessgwas.sh --plinkFile=$proj_dir/raw/$plink_name.rmoutlier --ldR2=0.02"
echo 
} >> "${file}"


# run the slurm script
sbatch -p "$node" "$file"
