#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --output=jupyter.log
# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
port=8889

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@hamilton.cs.mtsu.edu

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
module load miniconda
source activate my_env1
# Run Jupyter
jupyter lab --no-browser --port=${port} --ip=${node}
