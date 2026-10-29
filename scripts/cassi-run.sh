#!/bin/bash

# This script uses CASSI software to compute SNP-SNP interaction and then converts it to BridGE's sparse interaction network format (.npz)
# Options:
#       - --plinkFile: input plink file without the extension
#       - --GI: genetic interaction type, similar to CASSI. Check CASSI for more information.
#       - --pvalue: p-value cutoff for filtering interactions in CASSI.
#       - --nSample: number of sample permutations to be used for generating random interaction networks.
#       - --id: random network number (0 means real network) to be computed. This can be used instead of --nSample for creating only one network.
# https://www.staff.ncl.ac.uk/richard.howey/cassi/


# parsing parameters
for i in "$@"; do
        case $i in
                --plinkFile=*)
                        infile="${i#*=}"
                        shift
                        ;;
                --GI=*)
                        gi="${i#*=}"
                        shift
                        ;;
                --pvalue=*)
                        pval="${i#*=}"
                        shift
                        ;;
                --nSample=*)
                        n="${i#*=}"
                        shift
                        ;;
                --id=*)
                        net_id="${i#*=}"
                        shift
                        ;;
                --*)
                        echo "Unknown option $i"
                        exit 1
                        ;;
                *)
                        ;;
        esac
done

if [ -z "$infile" ]; then
        echo "plink input file not provided";
        exit 1;
fi

if [ -z "$gi" ]; then
        echo "CASSI interaction type not provided";
        exit 1;
fi

if [ -z "$pval" ]; then
        echo "p-value cutoff not provided";
        exit 1;
fi

if [ -z "$n" ] && [ -z "$net_id" ] ; then
        echo "number of sample permutations and network id not provided";
        exit 1;
fi



# plink files live in <project>/raw; networks go to <project>/intermediate, named like
# the ones bridge.py computes (snp_interaction_networks-<model>-R<i>.npz) with model
# cassi_<GI>. Pass them to bridge.py with --ssmFile=snp_interaction_networks-cassi_<GI>-R<i>.npz;
# ComputeFDR and Summarize take the -R0 file and find the pathway stats of R1..R<n> from it.
raw_dir="$(dirname "${infile}")"
intermediate_dir="$(dirname "${raw_dir}")/intermediate"
mkdir -p "${intermediate_dir}"


# Dciding on the command based on interaction type
case $gi in
        "lr")
                # for lr
                th_command="-lr -lr-th"
                model="cassi_LR"
                ;;
        "lin")
                th_command="-lin -lin-th"
                model="cassi_LIN"
                ;;
        "je")
                th_command="-je -je-thcc"
                model="cassi_JE"
                ;;
        "awu")
                th_command="-awu -awu-thcc"
                model="cassi_AWU"
                ;;
        "afe")
                th_command="-afe -afe-thcc"
                model="cassi_AFE"
                ;;
        "wz")
                th_command="-wz -wz-thcc"
                model="cassi_WZ"
                ;;
        *)
                echo "unknown CASSI interaction type: ${gi}"
                exit 1
                ;;
esac


# CASSI text output for network i, and the converted network
cassi_txt() { echo "${raw_dir}/${model}-R$1.txt"; }
network_npz() { echo "${intermediate_dir}/snp_interaction_networks-${model}-R$1.npz"; }


seed=6377474
if [ -z "$net_id" ] ; then 

        # Step 1, compute SNP-SNP interaction using CASSI
        cassi -i ${infile}.bed  ${th_command} ${pval} -max 0 -o $(cassi_txt 0)
        # Step 2, convert SNP-SNP interaction to BridGE's sparse network format
        python3 -m cassissm ${infile}.bim $(cassi_txt 0) ${gi} $(network_npz 0)
        for i in `seq 1 $n` ;
        do
                tmp_seed=$(($seed+$i))
                plink --bfile $infile --seed $tmp_seed --make-perm-pheno 1 --out ${raw_dir}/rand_pheno${i}
                plink --bfile $infile --pheno ${raw_dir}/rand_pheno${i}.pphe --mpheno 1 --make-bed --out ${infile}_R${i}
                cassi -i ${infile}_R${i}.bed ${th_command} ${pval}  -max 0 -o $(cassi_txt $i)
                python3 -m cassissm ${infile}.bim $(cassi_txt $i) ${gi} $(network_npz $i)
                rm ${raw_dir}/rand_pheno${i}.pphe
        done
else
        # find which network is being created
        if [ "$net_id" == "0" ];
        then
                cassi -i ${infile}.bed  ${th_command} ${pval} -max 0 -o $(cassi_txt 0)
                python3 -m cassissm ${infile}.bim $(cassi_txt 0) ${gi} $(network_npz 0)
        else
                tmp_seed=$(($seed+$net_id))
                plink --bfile $infile --seed $tmp_seed --make-perm-pheno 1 --out ${raw_dir}/rand_pheno${net_id}
                plink --bfile $infile --pheno ${raw_dir}/rand_pheno${net_id}.pphe --mpheno 1 --make-bed --out ${infile}_R${net_id}
                cassi -i ${infile}_R${net_id}.bed ${th_command} ${pval}  -max 0 -o $(cassi_txt $net_id)
                python3 -m cassissm ${infile}.bim $(cassi_txt $net_id) ${gi} $(network_npz $net_id)
                rm ${raw_dir}/rand_pheno${net_id}.pphe
        fi

fi
