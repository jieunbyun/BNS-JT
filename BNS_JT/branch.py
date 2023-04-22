import numpy as np
import textwrap


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
{self.__class__.__name__}(down={self.down}, up={self.up}, is_complete={self.is_complete}''')


def get_cmat(branches, comp_var_idx, varis, flag_comp_state_order=True):
    """
    Parameters
    ----------
    branches:

    flag_comp_state_order: 1 (default) if bnb and mbn have the same component states, 0 if bnb has a reverse ordering of components being better and worse
    """
    complete_brs = [x for x in branches if x.is_complete]
    #nRow = length( complete_brs )
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
                b1[down_state-1:up_state-1] = 1

                loc = ismember(b1, b)[1]

                if any(loc):
                    # conversion to python index
                    c[j+1] = loc[0] + 1
                else:
                    b = np.vstack((b, b1))
                    varis[comp_var_idx[j]].B = b
                    c[j+1] = b.shape[1]
            else:
                c[j+1] = up_state

        C[irow,:] = c

    return C, varis
