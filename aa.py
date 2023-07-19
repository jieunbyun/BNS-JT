import os
import socket
import time
import logging
import numpy as np
import pdb
import json
from pathlib import Path
from dask.distributed import Client, LocalCluster

import dask
from dask.distributed import Client,LocalCluster
#from dask_jobqueue import PBSCluster

from argparse import ArgumentParser
from BNS_JT import config, variable, model, cpm, branch, trans

HOME = Path(__file__).absolute().parent

#def main(client_ip=None):
def main(client):
    """
    walltime = '01:00:00'
    cores = 112
    memory = '192GB'

    cluster = PBSCluster(walltime=str(walltime), cores=cores, memory=str(memory),processes=cores,
                     job_extra_directives=['-q normalbw','-P n74','-l ncpus='+str(cores),'-l mem='+str(memory),
                                '-l storage=scratch/y57+scratch/n74+gdata/n74'],
                     local_directory='$TMPDIR',
                     header_skip=["select"],
                     )
    cluster.scale(jobs=10)

    """
    #cluster = LocalCluster(n_workers=10, threads_per_worker=10, memory_limit='64GB')
    # normalbw: 56, 192GB => workers:8, threads: 7, mem:96GB
    #cluster = LocalCluster()
    #cluster = LocalCluster(n_workers=112, threads_per_worker=1, memory_limit='32GB')
    cfg_file = HOME.joinpath('BNS_JT/demos/SF/config_SF.json')
    cfg = config.Config(cfg_file)
    #cfg = config.Config('./BNS_JT/tests/config_rbd.json')

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms = {}
    varis = {}

    # FIXME: only works for binary ATM
    B = np.vstack([np.eye(cfg.no_ds), np.ones(cfg.no_ds)])

    # only value is related to the scenario 
    s1 = list(cfg.scenarios['scenarios'].keys())[0]
    for k, values in cfg.scenarios['scenarios'][s1].items():
        varis[k] = variable.Variable(name=k, B=B, values=cfg.scenarios['damage_states'])
        cpms[k] = cpm.Cpm(variables = [varis[k]],
                no_child = 1,
                C = np.arange(len(values))[:, np.newaxis],
                p = values)


    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')


    # FIXME: only works for binary ATM
    lower = {k: 0 for k, _ in cfg.infra['edges'].items()}
    upper = {k: 1 for k, _ in cfg.infra['edges'].items()}

    variables = {k: varis[k] for k in cfg.infra['edges'].keys()}
    for k, v in cfg.infra['ODs'].items():
        k, v

    values = [np.inf] + sorted([y for _, y in path_times[v]], reverse=True)

    varis[k] = variable.Variable(name=k, B=np.eye(len(values)), values=values)

    #tic = time.time()
    _file = HOME.joinpath('BNS_JT/demos/SF/path_time_idx.json')
    with open(_file, 'r') as fid:
        path_time_idx_dic = json.load(fid)

    path_time_idx = path_time_idx_dic['od1']

    #path_time_idx = trans.get_path_time_idx(path_times[v], varis[k])
    #print(time.time() - tic)
    """
    dask.config.set({'logging.distributed.client': 'error',
                     'logging.distributed.scheduler': 'error',
                     'logging.distributed.nanny': 'error',
                     'logging.distributed.worker': 'error',
                     'logging.distributed.utils_perf': 'error'})
    """
    # FIXME
    tic = time.time()
    #pdb.set_trace()
    print(client)
    sb = branch.branch_and_bound_dask(path_time_idx, lower, upper, 1, client)
    print(time.time() - tic)

    """
    tic = time.time()
    c = branch.get_cmat_from_branches(sb, variables)
    print(time.time() - tic)
    np.savetxt('./c.txt', c, fmt='%d')
        cpms[k] = cpm.Cpm(variables = [varis[k]] + list(variables.values()),
               no_child = 1,
               C = c,
               p = np.ones(c.shape[0]),
               )
    """
def process_commandline():
    parser = ArgumentParser()
    #
    #parser.add_argument("-c", "--config",
    #                    dest="config_file",
    #                    help="read configuration from FILE",
    #                    metavar="FILE")
    parser.add_argument("-i", "--ip",
                        dest="client_ip",
                        help="set client ip address for dask cluster",
                        metavar="ip_address")
    return parser


if __name__=='__main__':
    #parser = process_commandline()
    #args = parser.parse_args()
    #main(client_ip=args.client_ip)

    #if 'gadi' in socket.gethostname():
    #client = Client(scheduler_file=os.environ["DASK_PBS_SCHEDULER"])
    #else:
    try:
        client = Client(scheduler_file=os.environ["DASK_PBS_SCHEDULER"])
    except KeyError:
        cluster = LocalCluster(n_workers=28, threads_per_worker=2, memory_limit='64GB')
        client = Client(cluster)

    main(client)
