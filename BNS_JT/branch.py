import numpy as np
import textwrap
import json
import time
import copy
import gc
import pdb
import itertools
from pathlib import Path
#import dask
#from dask.distributed import Client, worker_client, as_completed, get_client

#import dask.bag as db
from BNS_JT import cpm, variable, trans


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

    def __init__(self, down, up, names, is_complete=False, down_state=1, up_state=1, down_val=None, up_val=None):

        self.down = down
        self.up = up
        self.is_complete = is_complete
        self.down_state = down_state
        self.up_state = up_state
        self.down_val = down_val
        self.up_val = up_val
        self.names = names
        assert isinstance(down, list), 'down should be a list-like'

        assert isinstance(up, list), 'down should be a list-like'

        assert isinstance(names, list), 'names should be a list'

        assert len(down) == len(up), 'Vectors "down" and "up" must have the same length.'

        assert len(down) == len(names), 'Vectors "down/up" and "names" must have the same length.'

        assert isinstance(is_complete, bool), '"is_complete" must be either true (or 1) or false (or 0)'

        assert isinstance(down_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

        assert isinstance(up_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

    def __repr__(self):
        return textwrap.dedent(f'''\
{self.__class__.__name__}(down={self.down}, up={self.up}, is_complete={self.is_complete}, down_state={self.down_state}, up_state={self.up_state}, down_val={self.down_val}, up_val={self.up_val}''')

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Branch):
            return all([self.down == other.down,
                        self.up == other.up,
                        self.names == other.names,
                        self.is_complete == other.is_complete,
                        self.down_state == other.down_state,
                        self.up_state == other.up_state])

        return NotImplemented


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

                _, loc = cpm.ismember(b1, b)

                if any(loc):
                    # conversion to python index
                    c[j + 1] = loc[0]
                else:
                    print(f'B of {comp_var[j].name} is updated')
                    b = np.vstack((b, b1))
                    comp_var[j] = variable.Variable(name= comp_var[j].name,
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
                       is_complete=False,
                       names=info['arcs'])]

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


def get_sb_saved_from_job(output_path, key):
    '''
    output_path: str or Path
    '''
    try:
        assert output_path.exists(), f'output_path does not exist'
    except AttributeError:
        output_path = Path(output_path)
        assert output_path.exists(), f'output_path does not exist'
    finally:
        # read sb_saved json
        sb_saved = []
        for x in output_path.glob(f'sb_dump_{key}_*.json'):
            with open(x, 'r') as fid:
                tmp = json.load(fid)
                [sb_saved.append(tuple(x)) for x in tmp if x[2] == x[3]]

    return sb_saved


def get_bstars_from_sb_dump(file_name):

    bstars = []
    with open(file_name, 'r') as fid:
        tmp = json.load(fid)
        [bstars.append(tuple(x)) for x in tmp if x[2] != x[3]]

    return bstars


def get_sb_saved_from_sb_dump(file_name):

    sb_saved = []
    with open(file_name, 'r') as fid:
        tmp = json.load(fid)
        [sb_saved.append(tuple(x)) for x in tmp if x[2] == x[3]]

    return sb_saved


def split(list_a, chunk_size):

  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]


def branch_and_bound(bstars, path_time_idx, arc_cond, output_path, key):
    """
    client:
    bstars:
    path_time_idx:
    g_arc_cond:
    """

    i=0
    while bstars:

        print(f'{i}, b*: {len(bstars)}')

        tic = time.perf_counter()

        results = bnb_core(bstars, path_time_idx, arc_cond)

        bstars = [x for result in results for x in result if x not in bstars]

        output_file = output_path.joinpath(f'sb_dump_{key}_{i}.json')
        with open(output_file, 'w') as w:
            json.dump(bstars, w, indent=4)

        print(f'elapsed {i}: {time.perf_counter()-tic:.5f}')

        # next iteration
        bstars = [x for x in bstars if x[2] != x[3]]
        i += 1


def bnb_core(bstars, path_time_idx, arc_cond):
    """
    return a list of set of branches given bstars
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    g_arc_cond: arc_cond dask global variable
    """

    results = []
    for bstar in bstars:
        arcs = get_arcs_given_bstar(bstar, path_time_idx, arc_cond)
        result = get_set_of_branches(bstar, arcs, path_time_idx, arc_cond)
        results.append(result)

    return results


def get_arcs_given_bstar(bstar, path_time_idx, arc_cond):

    lower, upper, _, _ = bstar

    upper_matched = [k for k, v in upper.items() if v == arc_cond]

    arcs = []
    for _path, _, _ in path_time_idx[1:]:

        if set(_path).issubset(upper_matched):

            arcs = [x for x in _path if upper[x] > lower[x]]

            break

    return arcs


def get_arcs_given_bstar_nobreak(bstar, path_time_idx, arc_cond):

    lower, upper, _, _ = bstar

    upper_matched = [k for k, v in upper.items() if v == arc_cond]

    _path = next((x for x, _, _ in path_time_idx[1:] if set(x).issubset(upper_matched)), None)

    arcs = [x for x in _path if upper[x] > lower[x]]

    return arcs


def create_arc_state_given_cond(arc, **kwargs):
    """
    return dict of arc state given condition
    arc: str
    kwargs['arc_state']: base arc_state
    kwargs['value']: 0 or 1
    """

    return {k:kwargs['value'] if k==arc else v for k, v in kwargs['arc_state'].items()}


"""
def get_set_branches_no_iteration_(bstar, arcs, path_time_idx, arc_cond):

    lower, upper, c_fl, c_fu = bstar

    uppers = [{k:0 if k==arc else v for k, v in upper.items()} for arc in arcs]

    lowers = [lower]
    [lowers.append({k: 1 if k in arcs[:i] else v for k, v in lower.items()}) for i, _ in enumerate(arcs, 1)]

    fus = [trans.eval_sys_state_given_arc(upper, path_time_idx=path_time_idx, arc_cond=arc_cond) for upper in uppers]

    sb = [(lower, upper, fl, fu) for lower, upper, fl, fu in zip(lowers[:-1], uppers, [c_fl]*len(uppers), fus)]
    sb.append((lowers[-1], upper, c_fu, c_fu))

    return sb
"""


def get_set_of_branches(bstar, arcs, path_time_idx, arc_cond):

    lower, upper, c_fl, c_fu = bstar
    upper_f = {k: 1 if k in arcs else v for k, v in upper.items()}

    sb = []
    for arc in arcs:

        # set upper_n = 0
        #upper = {k: 0 if k == arc else v for k, v in upper.items()}
        upper = create_arc_state_given_cond(arc, value=0, arc_state=bstar[1])

        fu = trans.eval_sys_state_given_arc(upper, path_time_idx=path_time_idx, arc_cond=arc_cond)

        sb.append((lower, upper, c_fl, fu))

        # set upper_n=1, lower_n = 1 
        lower = create_arc_state_given_cond(arc, value=1, arc_state=lower)
        #lower = {k: 1 if k == arc else v for k, v in lower.items()}

    sb.append((lower, upper_f, c_fu, c_fu))

    return sb


def branch_and_bound_org(bstars, path_time_idx, arc_cond):
    """
    path_time_idx: a list of tuples consisting of path, time, and index (corresponding to row of B matrix)
    lower:
    upper:
    arc_cond:
    """

    #fl = trans.eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    #fu = trans.eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    #sb = [(lower, upper, fl, fu)]

    # selecting a branch from sb such that fl /= fu
    #bstars = [x for x in sb if x[2] != x[3]]

    # make sure the paths are sorted by shortest
    #paths_avail = [x[0] for x in path_time_idx if x[0]]
    sb_saved = []
    while bstars:

        print(f'b*: {len(bstars)}')
        # select path using upper branch of bstar
        for bstar in bstars:

            c_lower, c_upper, c_fl, c_fu = bstar
            upper_matched = [k for k, v in c_upper.items() if v == arc_cond]

            for _path, _, _ in path_time_idx[1:]:

                if set(_path).issubset(upper_matched):

                    upper = c_upper
                    lower = c_lower
                    fl = c_fl
                    chosen = (c_lower, c_upper, c_fl, c_fu)
                    sb = [x for x in bstars if not x == chosen]

                    #paths_avail.remove(_path)
                    break

            for arc in _path:

                if c_upper[arc] > c_lower[arc]:

                    # set upper_n = 0
                    upper = {k: 0 if k == arc else v for k, v in upper.items()}
                    #upper = copy.deepcopy(upper)
                    #upper[arc] = 0
                    fu = trans.eval_sys_state(path_time_idx, upper, arc_cond)
                    #fl = trans.eval_sys_state(path_time_idx, lower, arc_cond)
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
            #fl = trans.eval_sys_state(path_time_idx, lower, arc_cond)

            #if fl==fu:
                #sb.append((lower, upper, fl, fu))
            sb.append((lower, upper, c_fu, c_fu))

        bstars = [x for x in sb if x[2] != x[3]]
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

    C = C.astype(int)

    return C[C[:, 0].argsort()]


