#!/bin/sh
#SBATCH --job-name=BD5-parallel-experiment
#SBATCH -D /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/
#SBATCH -o /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/logs/%j.out
#SBATCH -e /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/logs/%j.err
#SBATCH -t 11:10:00
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH -p hpc5-el7-gpu-3d

export CUDA_HOME=/s/ls4/sw/cuda/10.1/
export LD_LIBRARY_PATH="/s/ls4/sw/cuda/10.1/lib64:$LD_LIBRARY_PATH"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/s/ls4/users/aamore/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/s/ls4/users/aamore/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/s/ls4/users/aamore/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/s/ls4/users/aamore/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


export PATH="/s/ls4/users/aamore/anaconda3/bin:$PATH"  # commented out by conda$



export PATH="/s/ls4/users/aamore/anaconda3/bin:$PATH"
echo `which python`

# echo `source /s/ls4/users/aamore/anaconda3/bin/activate pytorch_rl`
source /s/ls4/users/aamore/anaconda3/bin/activate pytorch_rl
# conda activate new_torch
echo `which python`
echo `which conda`
echo `conda env list`
echo ----------
conda activate new_torch

echo `which python`



param=$1
python /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/scripts/parallel_eps_runner.py  /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini ${param} | tee mytask_logs/mytask.log."$SLURM_JOBID"

