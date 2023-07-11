#PBS -m e
#PBS -P n74
#PBS -q normal 
#PBS -l walltime=12:00:00
#PBS -l ncpus=96
#PBS -l mem=32GB
#PBS -l wd
#PBS -N job
#PBS -l jobfs=1GB
#PBS -l storage=scratch/y57+scratch/n74

#module use /g/data3/hh5/public/modules
#module load conda/analysis3
#source activate wistl
#python -m wistl.main -c ./wistl/tests/test_memory.cfg &> test_memory.log
#./batch_dask.sh $PBS_NCPUS 2 &> batch_dask.log
./batch_dask.sh &> batch_dask.$PBS_JOBID.log
