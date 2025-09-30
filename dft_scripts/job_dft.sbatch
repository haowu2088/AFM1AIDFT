#!/bin/bash                     
#SBATCH --partition=768g             
#SBATCH --nodelist=execute-[702]    
#SBATCH --ntasks=24                  
#SBATCH --mem-per-cpu=30G          
#SBATCH --cpus-per-task=1    
#SBATCH --output=/dev/null       # Suppress default Slurm output logs 
#SBATCH --job-name=22679          # Input                               

########################################################################################################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Setting OpenMP threads to match the number of CPUs per task  

# Set ROOT_DIR based on the partition used
if [[ "$SLURM_JOB_PARTITION" == "dell" ]]; then
    ROOT_DIR='/home/agterber/wu67'
elif [[ "$SLURM_JOB_PARTITION" == "thinkpad" ]]; then
    ROOT_DIR='/home/haowu/wu67'
else
    ROOT_DIR='/home/uwm/wu67/Data/wu67'
fi
export ROOT_DIR  # Export to make it available to Python script

export STRUCTURE_CHOICE="atoms"   # "atoms" , "supercell"

export ECUT="100"
export RF="12"
export NK="12"
export DEGAUSS="0.01"
export CONV_THR="1e-6"
export DEGAUSS_INSULATOR="1e-9"
export MIXING_BETA="0.7"
export STEP="300"

########################################################################################################################
source ${ROOT_DIR}/conda/bin/activate  
conda activate dft1-env   

########################################################################################################################
# Enter in the terminal: example: sbatch job.sbatch mp-22679 FM
material_id=$1                 # sbatch terminal argument, eg., mp-13
magnetic_structure=$2            # sbatch terminal argument, eg., AFM1, FM, AFM2, AFM3

python_script="qe_test.py"    # INPUT Hardcoded parameter  "qe_test.py", "qe_single.py", "qe_relax.py", "qe_vcrelax.py"
test_parameter="ecut"      # INPUT  Hardcoded parameter "ecut", "rf", "nk", "degauss", "conv_thr", "special"

mkdir -p "output/$SLURM_JOB_NAME"  # -p creates the directory if it doesn't exist, without errors if it already does.
output_log="output/$SLURM_JOB_NAME/job${SLURM_JOB_ID}_${material_id}_${magnetic_structure}.txt"

# {...} &> "${output_log}" logs all information into output_log
# > writes fresh, >> appends, and & ensures both stdout and stderr are logged.
{
start_time=$(date +%s)
if [[ "${python_script}" == *"test"* ]]; then
    echo "Running ${material_id} ${magnetic_structure} ${python_script} ${test_parameter}"
    python ${python_script} ${material_id} ${magnetic_structure} ${test_parameter}
else
    echo "Running ${material_id} ${magnetic_structure} ${python_script}"
    python ${python_script} ${material_id} ${magnetic_structure}
fi
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$((total_time % 60))
echo "Total Calculation Time: ${hours}h ${minutes}m ${seconds}s"
echo -e "\nJob Done!"  # -e is for \n effect
} &> "${output_log}"     

########################################################################################################################
# Save this to indicate the Slurm file size is large upon completion, so I know the job finished without opening it.
{
echo -e "\n\n\n===== Sbatch Job Script ====="
cat $0  # $0 is the current Slurm script
echo -e "\n\n===== Python Script: $python_script ====="
cat $python_script
} &>> "${output_log}"  


########################################################################################################################
# Sleeping for 10 seconds to ensure stability
sleep 10
