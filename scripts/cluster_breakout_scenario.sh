#!/bin/sh
#SBATCH -D /s/ls4/users/aamore/2level_ConvHACO/
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p hpc5-gpu-3d


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

echo `source /s/ls4/users/aamore/anaconda3/bin/activate pytorch_rl`
echo `which python`
echo `which conda`
echo `conda env list`
echo ----------
conda activate pytorch_rl

python ./several_eps_runner.py ./configs/breakout_base_reborn.ini | tee mytask.log."$SLURM_JOBID"
