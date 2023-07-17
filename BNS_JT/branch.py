import numpy as np
import textwrap
import json
import time
import copy
import gc
import pdb
from pathlib import Path
import dask
from dask.distributed import Client, worker_client

from BNS_JT.cpm import ismember
from BNS_JT.variable import Variable
from BNS_JT.trans import eval_sys_state

class Branch(object):
    """

    Parameters
    ----------
    down
    up
    isComplete=false # 0 unknown, 1 confirmed
    down_state # associated system state on the lower bound (if 0, unknown)
    up_state # associated system state on the upper bound (if 0, unknown)
    down_val # (optional) a representative value of an associated state
    up_val # (optional) a representative value of an associated state
    """

    def __init__(self, down, up, is_complete=False, down_state=1, up_state=1, down_val=None, up_val=None):

        self.down = down
        self.up = up
        self.is_complete = is_complete
        self.down_state = down_state
        self.up_state = up_state
        self.down_val = down_val
        self.up_val = up_val

        assert isinstance(down, list), 'down should be a list-like'

        assert isinstance(up, list), 'down should be a list-like'

        assert len(down) == len(up), 'Vectors "down" and "up" must have the same length.'

        assert isinstance(is_complete, bool), '"is_complete" must be either true (or 1) or false (or 0)'

        assert isinstance(down_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

        assert isinstance(up_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

    def __repr__(self):
        return textwrap.dedent(f'''\
{self.__class__.__name__}(down={self.down}, up={self.up}, is_complete={self.is_complete}, down_state={self.down_state}, up_state={self.up_state}, down_val={self.down_val}, up_val={self.up_val}''')


def get_cmat(branches, comp_var, flag=True):
    """
    Parameters
    ----------
    branches:
    comp_var:
    varis:
    flag: 1 (default) if bnb and mbn have the same component states, 0 if bnb has a reverse ordering of components being better and worse

    """
    assert isinstance(branches, list), 'branches must be a list'
    assert isinstance(comp_var, (list, np.ndarray)), 'comp_var must be a list-like'
    #assert isinstance(varis, dict), 'varis must be a dict'
    assert isinstance(flag, bool), 'flag should be either 0 or 1'
    #assert set(comp_var).difference(varis.keys()) == set(), 'varis should contain index of comp_var: {comp_var}'

    complete_brs = [x for x in branches if x.is_complete]

    #FIXME: no_comp = len(comp_var) instead?
    no_comp = len(complete_brs[0].down)

    C = np.zeros((len(complete_brs), no_comp + 1))

    for irow, br in enumerate(complete_brs):

        c = np.zeros(no_comp + 1)

        # System state
        c[0] = br.up_state

        # Component states
        for j in range(no_comp):
            down = br.down[j]
            up = br.up[j]

            b = comp_var[j].B
            no_state = b.shape[1]

            if flag:
                down_state = down - 1
                up_state = up - 1
            else:
                down_state = no_state - up
                up_state = no_state - down

            if up_state != down_state:
                b1 = np.zeros((1, b.shape[1]))
                b1[int(down_state):int(up_state)] = 1

                _, loc = ismember(b1, b)

                if any(loc):
                    # conversion to python index
                    c[j + 1] = loc[0]
                else:
                    print(f'B of {comp_var[j].name} is updated')
                    b = np.vstack((b, b1))
                    comp_var[j] = Variable(name= comp_var[j].name,
                                           B=b,
                                           values=comp_var[j].values)
                    c[j + 1] = b.shape[1]
            else:
                c[j + 1] = up_state

        C[irow, :] = c

    return C


def get_idx(x, flag=False):

    assert isinstance(x, list), 'should be a list'
    assert all([isinstance(y, Branch) for y in x]), 'should contain an instance of Branch'
    return [i for i, y in enumerate(x) if y.is_complete is flag]


def run_bnb(sys_fn, next_comp_fn, next_state_fn, info, comp_max_states):
    """
    return branch
    Parameters
    ----------
    sys_fn
    next_comp_fn
    next_state_fn
    info
    comp_max_states: list-like
    """
    assert callable(sys_fn), 'sys_fn should be a function'
    assert callable(next_comp_fn), 'next_comp_fn should be a function'
    assert callable(next_state_fn), 'next_state_fn should be a function'
    assert isinstance(info, dict), 'info should be a dict'
    assert isinstance(comp_max_states, list), 'comp_max_states should be a list'

    init_up = comp_max_states
    init_down = np.ones_like(comp_max_states).tolist() # Assume that the lowest state is 1

    branches = [Branch(down=init_down,
                       up=init_up,
                       is_complete=False)]

    incmp_br_idx = get_idx(branches, False)

    while incmp_br_idx:
        branch_id = incmp_br_idx[0]
        _branch = branches[branch_id]
        down = _branch.down
        up = _branch.up

        down_state, down_val, down_res = sys_fn(down, info)
        up_state, up_val, up_res = sys_fn(up, info)

        if down_state == up_state:
            _branch.is_complete = True
            _branch.down_state = down_state
            _branch.up_state = up_state
            _branch.down_val = down_val
            _branch.up_val = up_val
            del incmp_br_idx[0]

        else:
            # matlab vs python index or not
            #FIXME
            cand_next_comp = [info['arcs'][i] for i, (x, y) in enumerate(zip(up, down)) if x > y]
            #cand_next_comp = [i+1 for i, (x, y) in enumerate(zip(up, down)) if x > y]

            next_comp = next_comp_fn(cand_next_comp, down_res, up_res, info)
            #FIXME
            next_comp_idx = info['arcs'].index(next_comp)
            next_state = next_state_fn(next_comp,
                                       [_branch.down[next_comp_idx], _branch.up[next_comp_idx]],
                                       down_res,
                                       up_res,
                                       info)
            branch_down = copy.deepcopy(_branch)
            branch_down.up[next_comp_idx] = next_state

            branch_up = copy.deepcopy(_branch)
            branch_up.down[next_comp_idx] = next_state + 1

            del branches[branch_id]

            branches.append(branch_down)
            branches.append(branch_up)

            incmp_br_idx = get_idx(branches, False)

    return branches


def get_branch_given_paths(path, lower, upper, path_time_idx, arc_cond):
    """
    path: list of edges
    """
    sb = []
    for arc in path:

        # set un = 0
        upper = {k: 0 if k == arc else v for k, v in upper.items()}
        fu = eval_sys_state(path_time_idx, upper, arc_cond)
        fl = eval_sys_state(path_time_idx, lower, arc_cond)

        sb.append((lower, upper, fl, fu))

        # set un=1, ln = 1 
        upper = {k: 1 if k == arc else v for k, v in upper.items()}
        lower = {k: 1 if k == arc else v for k, v in lower.items()}

        fu = eval_sys_state(path_time_idx, upper, arc_cond)
        fl = eval_sys_state(path_time_idx, lower, arc_cond)

        sb.append((lower, upper, fl, fu))

    return sb



def branch_and_bound_old(path_time_idx, lower, upper, arc_cond):
    """
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    lower:
    upper:
    arc_cond:
    """

    fl = eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    sb = [(lower, upper, fl, fu)]

    for _path, _time, idx in path_time_idx:

        if _path:

            # select lower and upper branch from b_star
            for c_lower, c_upper, c_fl, c_fu in sb:

                upper_matched = [k for k, v in c_upper.items() if v == arc_cond]

                # selecting a branch from sb such that fl /= fu
                if (c_fl != c_fu) and set(_path).issubset(upper_matched):

                    upper = c_upper
                    lower = c_lower
                    fl = c_fl
                    chosen = (c_lower, c_upper, c_fl, c_fu)
                    sb = [x for x in sb if not x == chosen]
                    break

            for arc in _path:

                # set upper_n = 0
                upper = {k: 0 if k == arc else v for k, v in upper.items()}
                fu = eval_sys_state(path_time_idx, upper, arc_cond)

                sb.append((lower, upper, fl, fu))

                # set upper_n=1, lower_n = 1 
                upper = {k: 1 if k == arc else v for k, v in upper.items()}
                lower = {k: 1 if k == arc else v for k, v in lower.items()}

                fu = eval_sys_state(path_time_idx, upper, arc_cond)
                fl = eval_sys_state(path_time_idx, lower, arc_cond)

                # FIXME!!  (different from the logic)
                if fu==fl:
                    sb.append((lower, upper, fl, fu))

    return sb


def get_path_given_b_star(_b_star, arc_cond, path_time_idx):

    #_, c_upper, _, _ = _b_star

    upper_matched = [k for k, v in _b_star[1].items() if v == arc_cond]

    for _path, _, _ in path_time_idx[1:]:

        if set(_path).issubset(upper_matched):

            #upper = c_upper
            #lower = c_lower
            #fl = c_fl
            #chosen = (c_lower, c_upper, c_fl, c_fu)
            #sb = [x for x in b_star if not x == chosen]

            #paths_avail.remove(_path)
            break

    return _path


def fn_dummy(_b_star, _path, arc_cond, path_time_idx):

    #_path = get_path_given_b_star(_b_star, arc_cond, path_time_idx)

    c_lower, c_upper, c_fl, c_fu = _b_star
    upper = c_upper
    lower = c_lower

    sb = []
    for arc in _path:

        if c_upper[arc] > c_lower[arc]:

            # set upper_n = 0
            upper = {k: 0 if k == arc else v for k, v in upper.items()}
            fu = eval_sys_state(path_time_idx, upper, arc_cond)
            #fl = eval_sys_state(path_time_idx, lower, arc_cond)
            #assert fl == c_fl, f'fl is not equal to c_fl: {fl} vs {c_fl}'
            sb.append((lower, upper, c_fl, fu))

            # set upper_n=1, lower_n = 1 
            upper = {k: 1 if k == arc else v for k, v in upper.items()}
            lower = {k: 1 if k == arc else v for k, v in lower.items()}

    sb.append((lower, upper, c_fu, c_fu))

    return sb



def branch_and_bound_dask(path_time_idx, lower, upper, arc_cond, client, key=''):
    """
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    lower:
    upper:
    arc_cond:
    """

    #client = Client(client_ip)
    print(client.dashboard_link)
    #n_workers=4, threads_per_worker=1, processes=False)
    fl = eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    sb = [(lower, upper, fl, fu)]

    # selecting a branch from sb such that fl /= fu
    b_star = [x for x in sb if x[2] != x[3]]

    # make sure the paths are sorted by shortest
    #paths_avail = [x[0] for x in path_time_idx if x[0]]
    i = 0
    while b_star:
        print(f'b*: {len(b_star)}')
        tic = time.time()

        # select path using upper branch of b_star
        results = []

        #with worker_client() as client:
        for _b_star in b_star:
            #scattered_path = client.scatter(path_time_idx)
            _path = get_path_given_b_star(_b_star, arc_cond, path_time_idx)
            result = fn_dummy(_b_star, _path, arc_cond, path_time_idx)
            client.run(gc.collect)
            results.append(result)

        results = client.gather(results)
        [sb.append(x) for result in results for x in result if not x in sb]

        sb = [x for x in sb if not x in b_star]
        b_star = [x for x in sb if x[2] != x[3]]
        sb_saved = [x for x in sb if x[2] == x[3]]

        with open(f'sb_saved_{key}{i}.json', 'w') as w:
            json.dump(sb_saved, w, indent=4)
            i += 1
            sb = []
        toc = print(f'elapsed: {time.time()-tic}')

    # read sb_saved json
    sb_saved = []
    for x in Path().glob(f'sb_saved_{key}*.json'):
        with open(x, 'r') as fid:
            tmp = json.load(fid)
            [sb_saved.append(tuple(x)) for x in tmp]

    return sb_saved


def branch_and_bound_using_fn(path_time_idx, lower, upper, arc_cond):
    """
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    lower:
    upper:
    arc_cond:
    """

    fl = eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    sb = [(lower, upper, fl, fu)]

    # selecting a branch from sb such that fl /= fu
    b_star = [x for x in sb if x[2] != x[3]]

    # make sure the paths are sorted by shortest
    sb_saved = []
    while b_star:

        print(f'b*: {len(b_star)}, sb: {len(sb_saved)}')
        # select path using upper branch of b_star
        #chosen = []
        for _b_star in b_star:

            _path = get_path_given_b_star(_b_star, arc_cond, path_time_idx)
            _sb = fn_dummy(_b_star, _path, arc_cond, path_time_idx)
            #if fl==fu:
                #sb.append((lower, upper, fl, fu))
            #sb.append((lower, upper, c_fu, c_fu))
            [sb.append(x) for x in _sb if not x in sb]
            #chosen.append(_chosen)

        sb = [x for x in sb if not x in b_star]
        b_star = [x for x in sb if x[2] != x[3]]
        [sb_saved.append(x) for x in sb if x[2] == x[3]]
        sb = []

    return sb_saved


def branch_and_bound(path_time_idx, lower, upper, arc_cond):
    """
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    lower:
    upper:
    arc_cond:
    """

    fl = eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    sb = [(lower, upper, fl, fu)]

    # selecting a branch from sb such that fl /= fu
    b_star = [x for x in sb if x[2] != x[3]]

    # make sure the paths are sorted by shortest
    #paths_avail = [x[0] for x in path_time_idx if x[0]]
    sb_saved = []
    while b_star:

        print(f'b*: {len(b_star)}, sb: {len(sb)}')
        # select path using upper branch of b_star
        for _b_star in b_star:

            c_lower, c_upper, c_fl, c_fu = _b_star
            upper_matched = [k for k, v in c_upper.items() if v == arc_cond]

            for _path, _, _ in path_time_idx[1:]:

                if set(_path).issubset(upper_matched):

                    upper = c_upper
                    lower = c_lower
                    fl = c_fl
                    chosen = (c_lower, c_upper, c_fl, c_fu)
                    sb = [x for x in b_star if not x == chosen]

                    #paths_avail.remove(_path)
                    break

            for arc in _path:

                if c_upper[arc] > c_lower[arc]:

                    # set upper_n = 0
                    upper = {k: 0 if k == arc else v for k, v in upper.items()}
                    #upper = copy.deepcopy(upper)
                    #upper[arc] = 0
                    fu = eval_sys_state(path_time_idx, upper, arc_cond)
                    #fl = eval_sys_state(path_time_idx, lower, arc_cond)
                    # print(f'{fl} vs {c_fl}')

                    sb.append((lower, upper, fl, fu))

                    # set upper_n=1, lower_n = 1 
                    #upper = c_upper
                    upper = {k: 1 if k == arc else v for k, v in upper.items()}
                    lower = {k: 1 if k == arc else v for k, v in lower.items()}
                    #upper = copy.deepcopy(upper)
                    #upper[arc] = 1
                    #lower = copy.deepcopy(lower)
                    #lower[arc] = 1

            #fu = c_fu
            #fl = eval_sys_state(path_time_idx, lower, arc_cond)

            #if fl==fu:
                #sb.append((lower, upper, fl, fu))
            sb.append((lower, upper, c_fu, c_fu))

        b_star = [x for x in sb if x[2] != x[3]]
        [sb_saved.append(x) for x in sb if x[2] == x[3]]

    return sb_saved


def get_cmat_from_branches(branches, variables):
    """
    branches: list of tuples (lower, upper, fl, fu)
    variables: dict of instances of Variable
    """

    C = np.zeros(shape=(len(branches), len(variables) + 1))

    for i, (lower, upper, fl, fu) in enumerate(branches):

        assert fl == fu

        C[i, 0] = fl

        for j, (k, v) in enumerate(variables.items(), 1):

            joined = [x|y for x, y in zip(v.B[lower[k]], v.B[upper[k]])]

            irow = np.where((v.B==joined).all(axis=1))[0]

            assert irow.size==1

            C[i, j] = irow

    return C.astype(int)
