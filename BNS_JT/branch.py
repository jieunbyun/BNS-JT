import numpy as np
import textwrap
import json
import time
import copy
import gc
from collections import Counter
from itertools import chain
import pdb
import itertools
from pathlib import Path
from collections import namedtuple
from scipy.stats import beta
import pandas as pd
import pyarrow
#import dask
#from dask.distributed import Client, worker_client, as_completed, get_client

#import dask.bag as db
from BNS_JT import cpm, variable, trans, brc


#attr = ["down", "up", "down_state", "up_state", "p"]
#Branch = namedtuple("Branch", attr, defaults=(None,)*len(attr))

#Branch_p = namedtuple("Branch", ["down", "up", "down_state", "up_state", "p"])

def approx_prob_by_comps(down, up, probs):
    """

    """
    p = 1.0
    for k, v in down.items():
        p *= sum([probs[k][x] for x in range(v, up[k] + 1)])
    return p


class Branch(object):

    def __init__(self, down, up, down_state=None, up_state=None, p=None):

        assert isinstance(down, dict), 'down should be a dict'
        assert isinstance(up, dict), 'down should be a dict'
        assert len(down) == len(up), 'Vectors "down" and "up" must have the same length.'

        self.down = down
        self.up = up
        self.down_state = down_state
        self.up_state = up_state
        self.p = p

        #assert isinstance(is_complete, bool), '"is_complete" must be either true (or 1) or false (or 0)'

    def __repr__(self):
        #return textwrap.dedent(f"""\{self.__class__.__name__}(down={self.down}, up={self.up}, down_state={self.down_state}, up_state={self.up_state}, p={self.p}""")
        return (f"Branch(\n"
                f"  down={self.down},\n"
                f"  up={self.up},\n"
                f"  down_state='{self.down_state}',\n"
                f"  up_state='{self.up_state}',\n"
                f"  p={self.p}\n"
                f")")

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Branch):
            return all([self.down == other.down,
                        self.up == other.up,
                        self.down_state == other.down_state,
                        self.up_state == other.up_state,
                        self.p == other.p])

    def approx_prob(self, probs): # TODO: the naming is wrong, It should be "get_indep_prob" (the probability is not approximate when components are independent.)
        self.p = approx_prob_by_comps(self.down, self.up, probs)


    def get_compat_rules(self, rules):

        """
        lower: lower bound on component vector state in dictionary
               e.g., {'x1': 0, 'x2': 0 ... }
        upper: upper bound on component vector state in dictionary
               e.g., {'x1': 2, 'x2': 2 ... }
        rules: dict of rules
               e.g., {'s': [{'x1': 2, 'x2': 2}],
                      'f': [{'x1': 2, 'x2': 0}]}
        """
        assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

        compat_rules = {'s': [], 'f': []}

        for rule in rules['s']:
            if all([self.up[k] >= v for k, v in rule.items()]):
                c_rule = {k: v for k, v in rule.items() if v > self.down[k]}
                if c_rule:
                    compat_rules['s'].append(c_rule)

        for rule in rules['f']:
            if all([self.down[k] <= v for k, v in rule.items()]):
                c_rule = {k: v for k, v in rule.items() if v < self.up[k]}
                if c_rule:
                    compat_rules['f'].append(c_rule)

        return compat_rules
    
    def eval_state( self, rules ):

        """
        lower: lower bound on component vector state in dictionary
               e.g., {'x1': 0, 'x2': 0 ... }
        upper: upper bound on component vector state in dictionary
               e.g., {'x1': 2, 'x2': 2 ... }
        rules: dict of rules
               e.g., {'s': [{'x1': 2, 'x2': 2}],
                      'f': [{'x1': 2, 'x2': 0}]}
        """
        assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

        for rule in rules['s']:
            if self.up_state != 'u':
                break

            if all([self.up[k] >= v for k, v in rule.items()]):
                self.up_state = 's'


        for rule in rules['s']:
            if self.down_state != 'u':
                break

            if all([self.down[k] >= v for k, v in rule.items()]):
                self.down_state = 's'

        for rule in rules['f']:
            if self.up_state != 'u':
                break

            if all([self.up[k] <= v for k, v in rule.items()]):
                self.up_state = 'f'


        for rule in rules['f']:
            if self.down_state != 'u':
                break

            if all([self.down[k] <= v for k, v in rule.items()]):
                self.down_state = 'f'


    def approx_joint_prob_compat_rule(self, rule, rule_st, probs):
        assert isinstance(rule, dict), f'rule should be a dict: {type(rule)}'
        assert isinstance(rule_st, str), f'rule_st should be a string: {type(rule_st)}'
        assert isinstance(probs, dict), f'probs should be a dict: {type(probs)}'
        p = 1.0
        if rule_st == 's':
            for x, v in rule.items():
                p *= sum([probs[x][i] for i in range(v, self.up[x] + 1)])

        elif rule_st == 'f':
            for x, v in rule.items():
                p *= sum([probs[x][i] for i in range(self.down[x], v + 1)])

        return p


    def get_decomp_comp_using_probs(self, rules, probs):
        """
        rules: dict of rules
               e.g., {'s': [{'x1': 2, 'x2': 2}], 'f': [{'x3': 0}]}
        probs: dict
        """
        assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'
        assert isinstance(probs, dict), f'probs should be a dict: {type(probs)}'

        # get an order of component by their frequency in rules
        rules_st = [(k, x) for k, rule in rules.items() for x in rule]
        comps = Counter(chain.from_iterable([x[1] for x in rules_st]))
        comps = [x[0] for x in comps.most_common()]

        # get an order R by P  (higher to lower)
        if len(rules_st) > 1:
            rules_st = sorted(rules_st, key=lambda x: self.approx_joint_prob_compat_rule(x[1], x[0], probs), reverse=True)

        for st, rule in rules_st:
            for c in comps:
                state = None
                if c in rule and st == 's':
                    state = rule[c]

                elif c in rule and st == 'f':
                    state = rule[c] + 1

                try:
                    if self.down[c] < state and state <= self.up[c]:
                        return c, state

                except TypeError:
                    pass

    def get_new_branch(self, rules, probs, xd, xd_st, up_flag=True):
        """

        """
        if up_flag:
            up = self.up.copy()
            up[xd] = xd_st - 1
            up_st = brc.get_state(up, rules)

            down = self.down
            down_st = self.down_state

        else:
            up = self.up
            up_st = self.up_state

            down = self.down.copy()
            down[xd] = xd_st
            down_st = brc.get_state(down, rules)

        new_br = Branch(down=down, up=up, down_state=down_st, up_state=up_st)
        new_br.approx_prob(probs)

        return new_br

    def get_c(self, varis, st_br_to_cs):
        """
        return updated varis and state
        varis: a dictionary of variables
        st_br_to_cs: a dictionary that maps state in br to state in C matrix of a system event
        """
        names = list(self.up.keys())
        cst = np.zeros(len(names) + 1, dtype=int) # (system, compponents)

        if self.down_state == self.up_state:
            cst[0] = st_br_to_cs[self.down_state]
        else:
            cst[0] = st_br_to_cs['u']

        for i, x in enumerate(names):
            down = self.down[x]
            up = self.up[x]

            if up > down:
                states = list(range(down, up + 1))
                varis[x], st = variable.get_composite_state(varis[x], states)
            else:
                st = up

            cst[i + 1] = st

        return varis, cst


    def to_dict(self):
        return {'down': self.down, 'up': self.up, 'down_state': self.down_state, 'up_state': self.up_state, 'p': self.p}

    @staticmethod
    def from_dict(data):
        return Branch( down = data['down'], up = data['up'], down_state = data['down_state'], up_state = data['up_state'], p = data['p'] )

def save_brs_to_parquet(brs_list, file_path):
    """
    Save a list of Branch objects to a .parquet file with efficient handling
    of nested dictionaries by flattening them into columns.
    
    Parameters:
    brs_list (list): A list of Branch objects.
    file_path (str): The full path to save the .parquet file.

    Example usage (given that `branches` is a list of Branch objects):
    save_branches_to_parquet(branches, 'output_file.parquet')

    """
    # Create a list to hold flattened data
    data = []

    # Iterate over the branches to create a flattened dictionary for each Branch
    for br in brs_list:
        # Flatten the 'down' and 'up' dictionaries by prefixing keys for uniqueness
        flattened_data = {
            **{f"down_{k}": v for k, v in br.down.items()},
            **{f"up_{k}": v for k, v in br.up.items()},
            "down_state": br.down_state,
            "up_state": br.up_state,
            "p": br.p
        }
        data.append(flattened_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a .parquet file
    df.to_parquet(file_path, engine='pyarrow', index=False)



def load_brs_from_parquet(file_path):
    """
    Load a list of Branch objects from a .parquet file with flattened 'down' and 'up' data.
    
    Parameters:
    file_path (str): The path to the .parquet file.
    
    Returns:
    list: A list of Branch objects.
    """
    # Read the DataFrame from the .parquet file
    df = pd.read_parquet(file_path, engine='pyarrow')

    # Reconstruct the list of Branch objects
    brs_list = []
    for _, row in df.iterrows():
        # Reconstruct 'down' and 'up' dictionaries by filtering columns with appropriate prefixes
        down = {key.replace('down_', ''): row[key] for key in row.index if key.startswith('down_') and not key == 'down_state'}
        up = {key.replace('up_', ''): row[key] for key in row.index if key.startswith('up_') and not key == 'up_state'}

        # Create a Branch object with reconstructed dictionaries
        branch = Branch(
            down=down,
            up=up,
            down_state=row.get('down_state'),
            up_state=row.get('up_state'),
            p=row.get('p')
        )
        brs_list.append(branch)

    return brs_list


class Branch_old(object):
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

        #assert isinstance(down_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

        #assert isinstance(up_state, (int, np.int32, np.int64)), '"down_state" must be a positive integer (if to be input).'

    def __repr__(self):
        return textwrap.dedent(f"""\
{self.__class__.__name__}(down={self.down}, up={self.up}, is_complete={self.is_complete}, down_state={self.down_state}, up_state={self.up_state}, down_val={self.down_val}, up_val={self.up_val}""")

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Branch_old):
            return all([self.down == other.down,
                        self.up == other.up,
                        self.names == other.names,
                        self.is_complete == other.is_complete,
                        self.down_state == other.down_state,
                        self.up_state == other.up_state])

        return NotImplemented


def get_cmat(branches, comp_varis):
    """
    Parameters
    ----------
    branches: list of Branch
    comp_varis: dictionary of varis of component events

    Returns
    C: C matrix of the system event
        The first column is the system state (0: failure, 1: survival, 2: unknown)
    """
    assert isinstance(branches, list), 'branches must be a list'
    assert isinstance(comp_varis, dict), 'comp_var must be a dictionary'
    
    no_comp = len(comp_varis)

    C = np.zeros((0, no_comp + 1))

    for br in branches:

        c = np.zeros(no_comp + 1)
        c_comp = get_crow(br, comp_varis)
        c[1:] = c_comp

        if br.down_state == 's': # survival branch
            c[0] = 1
        elif br.up_state == 'f': # failure branch            
            c[0] = 0
        else: # unknown branch
            c[0] = 2 

        C = np.vstack((C, c))
        C = C.astype(int)

    return C

def get_crow(br1, comp_varis):

    c_comp = np.zeros((1, len(comp_varis)))
    
    for j, (k, v) in enumerate(comp_varis.items()):
        down = br1.down[k]
        up = br1.up[k]

        if up != down:
            bj = {x for x in range(int(down), int(up+1))}
            sj = v.get_state(bj)
            c_comp[0][j] = sj

        else:
            c_comp[0][j] = up
    
    return c_comp


def get_idx(x, flag=False):

    assert isinstance(x, list), 'should be a list'
    assert all([isinstance(y, Branch_old) for y in x]), 'should contain an instance of Branch'
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

    branches = [Branch_old(down=init_down,
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

            joined = v.B[lower[k]].union(v.B[upper[k]])

            #irow = v.B.index(joined)

            C[i, j] = v.B.index(joined)

    C = C.astype(int)

    return C[C[:, 0].argsort()]


def mcs_unknown_using_cpms(brs, probs, sys_fun, cpms, sys_name, cov_t, rand_seed=None):
    """
    Perform Monte Carlo simulation for the unknown state.

    INPUTS:
    brs_u: Unspecified branches (list)
    probs: a dictionary of failure probabilities for each component
    sys_fun: System function
    cpms: a list of cpms containing component events and system event
    sys_name: a string of the system event's name in cpms
    cov_t: a target c.o.v.
    rand_seed: Random seed

    OUTPUTS:
    cpms: cpms with samples added
    result: Results of the Monte Carlo simulation
    """

    brs_u = [b for b in brs if b.down_state == 'u' or b.up_state == 'u']
    pf_low = sum(b.p for b in brs if b.up_state == 'f')

    # Set the random seed
    if rand_seed:
        np.random.seed(rand_seed)

    brs_u_probs = [b.p for b in brs_u]
    brs_u_prob = sum(brs_u_probs)

    samples = []
    samples_sys = np.empty((0, 1), dtype=int)
    sample_probs = []

    nsamp, nfail = 0, 0
    pf, cov = 0.0, 1.0

    while cov > cov_t:

        nsamp += 1

        sample1 = {}
        s_prob1 = {}

        # select a branch
        br_id = np.random.choice(range(len(brs_u)), p=[p / brs_u_prob for p in brs_u_probs])
        br = brs_u[br_id]

        for e in br.down.keys():
            d = br.down[e]
            u = br.up[e]

            if d < u: # (fail, surv)
                st = np.random.choice(range(d, u + 1), p=[probs[e][d], probs[e][u]])
            else:
                st = d

            sample1[e] = st
            s_prob1[e] = probs[e][st]

        # system function run
        val, sys_st, _ = sys_fun(sample1)

        samples.append(sample1)
        sample_probs.append(s_prob1)
        if sys_st == 'f':
            samples_sys = np.vstack((samples_sys, [0]))
        else: # sys_st == 's'
            samples_sys = np.vstack((samples_sys, [1]))

        if sys_st == 'f':
            nfail += 1

        if nsamp > 9:
            prior = 0.01
            a, b = prior + nfail, prior + (nsamp-nfail) # Bayesian estimation assuming beta conjucate distribution

            pf_s = a / (a+b)
            var_s = a*b / (a+b)**2 / (a+b+1)
            std_s = np.sqrt(var_s)

            pf = pf_low + brs_u_prob *pf_s
            std = brs_u_prob * std_s
            cov = std/pf

            conf_p = 0.95 # confidence interval
            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = pf_low + brs_u_prob * np.array([low, up])

        if nsamp % 1000 == 0:
            print(f'nsamp: {nsamp}, pf: {pf:.4e}, cov: {cov:.4e}')

    # Allocate samples to CPMs
    Csys = np.zeros((nsamp, len(probs)), dtype=int)
    Csys = np.hstack((samples_sys, Csys))

    for i, v in enumerate(cpms[sys_name].variables[1:]):
        Cv = np.array([s[v.name] for s in samples], dtype=int).T
        cpms[v.name].Cs = Cv
        cpms[v.name].q = np.array([p[v.name] for p in sample_probs], dtype=float).T
        cpms[v.name].sample_idx = np.arange(nsamp, dtype=int)

        Csys[:, i+1] = Cv.flatten()

    cpms[sys_name].Cs = Csys
    cpms[sys_name].q = np.ones((nsamp,1), dtype=float)
    cpms[sys_name].sample_idx = np.arange(nsamp, dtype=int)

    result = {'pf': pf, 'cov': cov, 'nsamp': nsamp, 'cint_low': cint[0], 'cint_up': cint[1]}

    return cpms, result

def mcs_unknown_simple(brs, probs, sys_fun, cov_t, rand_seed=None):
    """
    Perform Monte Carlo simulation for the unknown state WITHOUT involving CPMs.
    (used when re-use of MCS samples is not considered)

    INPUTS:
    brs_u: Unspecified branches (list)
    probs: a dictionary of failure probabilities for each component
    sys_fun: System function
    cov_t: a target c.o.v.
    rand_seed: Random seed

    OUTPUTS:
    cpms: cpms with samples added
    result: Results of the Monte Carlo simulation
    """

    brs_u = [b for b in brs if b.down_state == 'u' or b.up_state == 'u']
    pf_low = sum(b.p for b in brs if b.up_state == 'f')

    # Set the random seed
    if rand_seed:
        np.random.seed(rand_seed)

    brs_u_probs = [b.p for b in brs_u]
    brs_u_prob = sum(brs_u_probs)

    nsamp, nfail = 0, 0
    pf, cov = 0.0, 1.0

    while cov > cov_t:

        nsamp += 1
        sample1 = {}

        # select a branch
        br_id = np.random.choice(range(len(brs_u)), p=[p / brs_u_prob for p in brs_u_probs])
        br = brs_u[br_id]

        for e in br.down.keys():
            d = br.down[e]
            u = br.up[e]

            if d < u: # (fail, surv)
                st = np.random.choice(range(d, u + 1), p=[probs[e][d], probs[e][u]])
            else:
                st = d

            sample1[e] = st

        # system function run
        val, sys_st, _ = sys_fun(sample1)

        if sys_st == 'f':
            nfail += 1

        if nsamp > 9:
            prior = 0.01
            a, b = prior + nfail, prior + (nsamp-nfail) # Bayesian estimation assuming beta conjucate distribution

            pf_s = a / (a+b)
            var_s = a*b / (a+b)**2 / (a+b+1)
            std_s = np.sqrt(var_s)

            pf = pf_low + brs_u_prob *pf_s
            std = brs_u_prob * std_s
            cov = std/pf

            conf_p = 0.95 # confidence interval
            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = pf_low + brs_u_prob * np.array([low, up])

        if nsamp % 1000 == 0:
            print(f'nsamp: {nsamp}, pf: {pf:.4e}, cov: {cov:.4e}')


    result = {'pf': pf, 'cov': cov, 'nsamp': nsamp, 'cint_low': cint[0], 'cint_up': cint[1]}

    return result