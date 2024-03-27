import numpy as np
from itertools import chain, combinations
import math

#from collections import namedtuple
#from typing import NamedTuple

"""
BaseVariable = namedtuple('BaseVariable', [
    'name',
    'B',
    'values'
    ])

class Variable(BaseVariable):
    '''
    A namedtuple subclass to hold Variable

    name: str
    B: basis set
    values: description of states

    B may be replaced by a dictionary something like "numBasicState = 3" and "compositeState = {'4':['1','2'], '5':['1','3'], '6': ['1','2','3']}"
    (A user does not have to enter composite states for all possible permutations but is enough define those being used).
    '''
    __slots__ = ()
    def __new__(cls, name, B, values):

        assert isinstance(name, str), 'name should be a string'

        assert isinstance(B, (np.ndarray, list)), 'B must be a array'

        if isinstance(B, list):
            B = np.array(B, dtype=int)

        if B.dtype == np.dtype(np.float64):
            B = B.astype(int)

        assert isinstance(values, list), 'values must be a list'

        numBasicState = B.shape[1]

        assert (B[:numBasicState, :] == np.eye(numBasicState)).all(), 'The upper part corresponding to basic states must form an identity matrix'

        return super(Variable, cls).__new__(cls, name, B, values)

    def B_times_values(self):

        return [' '.join(x).strip(' ') for x in np.char.multiply(self.values, self.B.astype(int)).tolist()]

    def __hash__(self):

        return hash(self.name)


    def __eq__(self, other):

        return self.name == other.name

    def __repr__(self):
        return repr(f'Variable(name={self.name}, B={self.B}, values={self.values})')
"""

class Variable(object):
    '''
    A namedtuple subclass to hold Variable

    name: str
    B: basis set
    values: description of states

    B may be replaced by a dictionary something like "numBasicState = 3" and "compositeState = {'4':['1','2'], '5':['1','3'], '6': ['1','2','3']}"
    (A user does not have to enter composite states for all possible permutations but is enough define those being used).
    '''

    def __init__(self, name, values=[]):

        assert isinstance(name, str), 'name should be a string'

        assert isinstance(values, list), 'values must be a list'

        self.name = name
        self._values = values

        ddd = 1
        
        #Do not create B explicitly
        """
        if self._values:
            self._B = self.gen_B()
        """

    @property
    def B(self):
        return self._B

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):

        assert isinstance(values, list), 'values must be a list'

        self._values = values

        #self._B = self.gen_B()

    def B(self, st=None):

        n = len(self._values)

        if st == None:
            B = [set(x) for x in chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))]
            return B
        
        else:
            assert isinstance(st, int) or isinstance(st, set), 'Given state must be an integer or a set'

            nst_len = [math.comb(n,r) for r in range(1,n+1)]  # number of states in each length from 1 to n
            nst_len_cum = np.cumsum(nst_len)

            #FIXME: still not passing tests
            if isinstance(st, int):
                
                st_len = np.argmax(~(nst_len_cum<st)) + 1 # length of the state
                
                st_to_go = st - ( 2**(st_len-1) - 1 )
                st_idx = 0
                for x in chain.from_iterable(combinations(range(n), st_len)):
                    if st_idx==st_to_go:
                        break
                    st_idx += 1
                return {x}
            
            else:
                
                #FIXME: still not passing tests
                st_len = len(st)
                st_idx = nst_len_cum[st_len-1]
                for x in chain.from_iterable(combinations(range(n), st_len)):
                    st_idx += 1
                    if set(x)==st:
                        break
                return st_idx



    """def gen_B(self):
        n = len(self._values)
        B = [set(x) for x in chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))]
        #for n in range(1, len(self._values) + 1):
        #    [B.append(set(x)) for x in itertools.combinations(range(len(self._values)), n)]
        return B"""

    """
    def B_times_values(self):

        return [' '.join(x).strip(' ') for x in np.char.multiply(self.values, self.B.astype(int)).tolist()]
    """


    #    self.check_B(value)
    #    self._B = value
    """
    def check_B(self, value):

        assert isinstance(value, list), 'B must be a list'
        assert len(value) >= len(self.values), 'B contains index or indices of the value'
        assert len(value) <= 2**len(self.values) - 1, f'Length of B can not exceed {2**len(self.values)-1}: {value}, {self.values}'
        assert all([max(v) < len(self.values) for v in value]), 'B contains index or indices of the value'
        assert all([isinstance(v, set) for v in value]), 'B consists of set'
    """

    def __hash__(self):

        return hash(self.name)

    def __lt__(self, other):

        return self.name < other.name

    def __eq__(self, other):

        return self.name == other.name

    def __repr__(self):
        return repr(f'Variable(name={self.name}, B={self.B}, values={self.values})')

#FIXME: seems obsolete
def get_composite_state(vari, states):
    """
    # Input: vari-one Variable object, st_list: list of states (starting from zero)
    # TODO: states start from 0 in Cpm and from 1 in B&B -- will be fixed later so that all start from 0

    b = [x in states for x in range(len(vari.B[0]))]

    comp_st = np.where((vari.B == b).all(axis=1))[0]

    if len(comp_st):
        cst = comp_st[0]

    else:
        vari.B = np.vstack((vari.B, b))
        cst = len(vari.B) - 1 # zero-based index
    """
    added = set(states)
    if added not in vari.B:
        vari.B.append(added)

    cst = vari.B.index(added)

    return vari, cst

