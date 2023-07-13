import time
import numpy as np
import pdb
from dask.distributed import Client, LocalCluster

from argparse import ArgumentParser
from BNS_JT import config, variable, model, cpm, branch
from Trans import trans


#def main(client_ip=None):
def main():

    cluster = LocalCluster()
    cfg = config.Config('./BNS_JT/demos/SF/config_SF.json')
    #cfg = config.Config('./Trans/tests/config_rbd.json')

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

        values = [np.inf] + sorted([y for _, y in path_times[v]], reverse=True)

        varis[k] = variable.Variable(name=k, B=np.eye(len(values)), values=values)

        tic = time.time()
        path_time_idx = trans.get_path_time_idx(path_times[v], varis[k])
        print(time.time() - tic)

        # FIXME
        tic = time.time()
        #pdb.set_trace()
        with Client(cluster) as client:
            sb = branch.branch_and_bound_dask(path_time_idx, lower, upper, 1, client)
        print(time.time() - tic)

        tic = time.time()
        c = branch.get_cmat_from_branches(sb, variables)
        print(time.time() - tic)
        np.savetxt('./c.txt', c, fmt='%d')
        """
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
    parser = process_commandline()
    args = parser.parse_args()
    #main(client_ip=args.client_ip)
    main()