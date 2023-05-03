import numpy as np
import textwrap
import copy


from BNS_JT.cpm import ismember


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


def get_cmat(branches, comp_var_idx, varis, flag_comp_state_order=True):
    """
    Parameters
    ----------
    branches:
    comp_var_idx:
    varis:
    flag_comp_state_order: 1 (default) if bnb and mbn have the same component states, 0 if bnb has a reverse ordering of components being better and worse

    """
    assert isinstance(branches, list), 'branches must be a list'
    assert isinstance(comp_var_idx, (list, np.ndarray)), 'comp_var_idx must be a list-like'
    assert isinstance(varis, dict), 'varis must be a dict'
    assert isinstance(flag_comp_state_order, bool), 'flag_comp_state_order should be either 0 or 1'
    assert set(comp_var_idx).difference(varis.keys()) == set(), 'varis should contain index of comp_var_idx: {comp_var_idx}'

    complete_brs = [x for x in branches if x.is_complete]

    #FIXME: no_comp = len(comp_var_idx) instead?
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

            b = varis[comp_var_idx[j]].B
            no_state = b.shape[1]

            if flag_comp_state_order:
                down_state = down
                up_state = up
            else:
                down_state = no_state + 1 - up
                up_state = no_state + 1 - down

            if up_state != down_state:
                b1 = np.zeros((1, b.shape[1]))
                b1[int(down_state)-1:int(up_state)-1] = 1

                _, loc = ismember(b1, b)

                if any(loc):
                    # conversion to python index
                    c[j + 1] = loc[0] + 1
                else:
                    #print(f'B of varis[{comp_var_idx[j]}] is updated')
                    b = np.vstack((b, b1))
                    varis[comp_var_idx[j]].B = b
                    c[j + 1] = b.shape[1]
            else:
                c[j + 1] = up_state

        C[irow, :] = c

    return C, varis


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
            next_comp_idx = np.where(info['arcs']==next_comp)[0][0] + 1
            next_state = next_state_fn(next_comp,
                                       [_branch.down[next_comp_idx], _branch.up[next_comp_idx]],
                                       down_res,
                                       up_res,
                                       info)
            branch_down = copy.deepcopy(_branch)
            branch_down.up[next_comp_idx - 1] = next_state

            branch_up = copy.deepcopy(_branch)
            branch_up.down[next_comp_idx - 1] = next_state + 1

            del branches[branch_id]

            branches.append(branch_down)
            branches.append(branch_up)

            incmp_br_idx = get_idx(branches, False)

    return branches


