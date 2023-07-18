#!/bin/bash
#PBS -N test
#PBS -P n74
#PBS -q normalbw
#PBS -l walltime=5:00:00
#PBS -l ncpus=112
#PBS -l mem=384GB
#PBS -l jobfs=100GB
#PBS -l storage=gdata/dk92+scratch/n74+gdata/n74

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-data-analysis/2023.02
module load gadi_jupyterlab/23.02
jupyter.ini.sh [ options ]
sleep infinity
