#PBS -M hyeuk.ryu@ga.gov.au
#PBS -m e
#PBS -P n74 
#PBS -q normalbw
#PBS -l walltime=10:02:00
#PBS -l ncpus=28
#PBS -l mem=256GB
#PBS -l wd
#PBS -N aa 
#PBS -l jobfs=20GB
#PBS -l storage=scratch/y57+scratch/n74+gdata/n74
#PBS -l other=hyperthread

rm -rf dask-worker-space

module load python3/3.10.4
export OMP_NUM_THREADS=$PBS_NCPUS
export PYTHONPATH=/g/data1b/n74/hxr547/bns/lib/python3.10/site-packages:$PYTHONPATH

python3 -u ./aa.py >& $PBS_JOBID.log

