set -eu

# The specific module+version that each node should load.
#module_name=$1
#shift
# ppn: processes per node: no. for worker processes to start on each node: no idea about process?
# tpp: threads per process (no. of threades for worker)
ppn=8
tpp=1

# this memory defines memory for worker process and it is limited to memory size in mem (PBS)
mem=4e9 # Four gigabytes per worker process
# in the log client processes = (ncpus-2), threads: (ncpus-2) * tpp, memory = mem(PBS) * ncpus
#INFO:<Client: 'tcp://10.6.24.71:45275' processes=6 threads=24, memory=3.15 GB>
umask=0027
HOME_PATH=/home/547/hxr547
WISTL_ENV_PATH=/scratch/n74/hxr547/pyinstall/lib/python3.10/site-packages
PROJ_PATH=$HOME_PATH/Projects/BNS-JT
export USER=hxr547
export PROJECT=n74

#while [[ $# -gt 0 ]]
#do
#    key="$1"
#    case $key in
#    --help)
#        #echo "Usage: $0 <dea_module> --umask ${umask} --ppn ${ppn} --tpp ${tpp} script args"
#        echo "Usage: $0 --ppn ${ppn} --tpp ${tpp} script args"
#        exit 0
#        ;;
#    --umask)
#        umask="$2"
#        shift
#        ;;
#    --ppn)
#        ppn="$2"
#        shift
#        ;;
#    --tpp)
#        tpp="$2"
#        shift
#        ;;
#    *)
#    break
#    ;;
#    esac
#shift
#done


#init_env="umask ${umask}; source /etc/bashrc; module use /g/data/v10/public/modules/modulefiles/; module use /g/data/v10/private/modules/modulefiles/; module load ${module_name}"
#init_env="umask ${umask}; source /etc/bashrc; source /home/547/hxr547/.bashrc; module use /g/data3/hh5/public/modules/; module load conda/analysis3; env"
#init_env="umask ${umask}; source /etc/bashrc; export USER=$USER; export PROJECT=$PROJECT; export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8; source /g/data3/hh5/public/apps/miniconda3/bin/activate /short/y57/hxr547/conda/envs/wistl"
init_env="umask ${umask}; source $HOME_PATH/.bash_profile; export USER=$USER; export PROJECT=$PROJECT; module load python3/3.10.4; export PYTHONPATH=/scratch/n74/hxr547/pyinstall/lib/python3.10/site-packages:/apps/python3/3.10.4/lib/python3.10/site-packages"
#init_env="umask ${umask}; source /etc/bashrc; source /home/547/hxr547/.bashrc; module use /g/data3/hh5/public/modules/; module load conda/analysis3; env"

#echo "Using DEA module: ${module_name}"

# Make lenient temporarily: global bashrc/etc can reference unassigned variables.
set +u
#eval "umask ${umask}"
#eval "source $HOME_PATH/.bash_profile"
#eval "export USER=$USER"
#eval "export PROJECT=$PROJECT"
#eval "module load python3/3.10.4; export PYTHONPATH=/scratch/n74/hxr547/pyinstall/lib/python3.10/site-packages:/apps/python3/3.10.4/lib/python3.10/site-packages"
#echo $PROJECT
eval "${init_env}"
set -u

SCHEDULER_NODE=$(sed '1q;d' "$PBS_NODEFILE")
SCHEDULER_PORT=$(shuf -i 2000-65000 -n 1)
SCHEDULER_ADDR=$SCHEDULER_NODE:$SCHEDULER_PORT

# node numbers: if ppn < NCPUS-2 then ppn, else NCPUS-2
# NCPUS: varies by queue (e.g., 2x14 for normalbw)
# PBS_NCPUS: requested by PBS job
n0ppn=$(( ppn < NCPUS-2 ? ppn : NCPUS-2 ))
n0ppn=$(( n0ppn > 0 ? n0ppn : 1 ))


# scheduler
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-scheduler --port $SCHEDULER_PORT"&
sleep 5s

# worker?
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${n0ppn} --nthreads ${tpp} --memory-limit ${mem} --name worker-0-$PBS_JOBNAME"&
sleep 1s

# worker: 
# no. of works = PBS_NCPUS/NCPUS 
for ((i=NCPUS; i<PBS_NCPUS; i+=NCPUS)); do
  echo "creating worker: ${i}"
  pbsdsh -n $i -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${ppn} --nthreads ${tpp} --memory-limit ${mem} --name worker-$i-$PBS_JOBNAME"&
  sleep 1s
done
sleep 5s

#datacube -vv system check
#set | grep -i datacube
# PBS_VMEM: requested memory in PBS
echo "
  #------------------------------------------------------
  # PBS Info
  #------------------------------------------------------
  # PBS: Qsub is running on $PBS_O_HOST login node
  # PBS: Originating queue      = $PBS_O_QUEUE
  # PBS: Executing queue        = $PBS_QUEUE
  # PBS: Working directory      = $PBS_O_WORKDIR
  # PBS: Execution mode         = $PBS_ENVIRONMENT
  # PBS: Job identifier         = $PBS_JOBID
  # PBS: Job name               = $PBS_JOBNAME
  # PBS: Node_file              = $PBS_NODEFILE
  # PBS: Current home directory = $PBS_O_HOME
  # PBS: PATH                   = $PBS_O_PATH
  # PBS: NCPUS                  = $PBS_NCPUS
  # PBS: NODENUM                = $PBS_NODENUM
  # PBS: MEM                    = $PBS_VMEM
  # PBS: PPN                    = $ppn
  # PBS: TPP                    = $tpp
  # DASK: NO_WORKERS            = $((PBS_NCPUS/NCPUS)) 
  #------------------------------------------------------"

#"${@/DSCHEDULER/${SCHEDULER_ADDR}}"
#echo $PROJ_PATH
cd $PROJ_PATH
echo "$SCHEDULER_ADDR"
python3 $PROJ_PATH/aa.py -i "${SCHEDULER_ADDR}" 
#python wistl/main.py -c $PROJ_PATH/wistl/tests/test1_parallel.cfg -i "${SCHEDULER_ADDR}"
#python wistl/aa.py
