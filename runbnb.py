import numpy as np
from BNS_JT.branch import Branch


def get_idx(x, flag=False):

    assert isinstance(x, list), 'should be a list'
    assert all([isinstance(y, Branch) for y in x]), 'should contain an instance of Branch'
    return [i for i, y in enmerate(x) if y.is_complete == flag]


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

    ncomp = len( comp_max_states )

    init_up = comp_max_states
    init_down = np.ones(nComp) # Assume that the lowest state is 1

    branches = [Branch(down=init_down,
                      up=init_up,
                      is_complete=False)]

    incmp_br_idx = get_idx_false(branches)

    while incomp_br_idx:

        branch_id = incmp_br_idx[0]
        _branch = branches[branch_id]
        down = _branch.down
        up = _branch.up

        down_state, down_val, down_res = sys_fn(down, up)
        up_state, up_val, up_res = sys_fn(up, info)

        if down_state == up_date:
            _branch.is_complete = True
            _branch.down_state = down_state
            _branch.up_state = up_state
            _branch.down_val = down_val
            _branch.up_val = up_val
            del incmp_br_idx[0]

        else:
            cand_next_comp = [i for i, (x, y) in enumerate(zip(up, down)) if x > y]
            next_comp = next_comp_fn(cand_next_comp, down_res, up_res, info)
            next_state = next_state_fn(next_comp,
                                       [_branch.down[next_comp], _branch.up[next_comp]],
                                       up_res, info)
            branch_down = _branch.copy()
            branch_down.up[next_comp] = next_state

            branch_up = _branch.copy()
            iBranch_up.down[next_comp] = next_state + 1

            del branches[branch_id]

            branches.append(branch_down)
            branches.append(branch_up)

            incmp_br_idx = get_idx(branches, False)

    return branches


