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

    def __init__(self, name, values=[], B_flag=None):

        assert isinstance(name, str), 'name should be a string'

        assert isinstance(values, list), 'values must be a list'

        self.name = name
        self.values = values
        self.B_flag = B_flag

        self.B = None
        if len(self.values) > 0 and ((len(self.values) <= 6 and B_flag != 'fly') or B_flag=='store'):
            self.B = self.gen_B()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):

        assert isinstance(values, list), 'values must be a list'

        self._values = values

        #self._B = self.gen_B()

    @property
    def B_flag(self):
        return self._B_flag

    @B_flag.setter
    def B_flag(self, value):

        assert value in [None, 'store', 'fly'], 'B_flag must be either None, store, or fly'
        self._B_flag = value

    def gen_B(self):
        n = len(self._values)
        B = [set(x) for x in chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))]
        #for n in range(1, len(self._values) + 1):
        #    [B.append(set(x)) for x in itertools.combinations(range(len(self._values)), n)]
        return B
            
    def get_state( self, set ):
        """
        Finds the state of a given set of basic states.

        The sets are ordered as follows:
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1}, {0, 1, 2}, ..., {0, 1, ..., n-1}]

        Parameters:
        set (set): The set of basic states for which to find the state.

        Returns:
        state: The state of the given set in the custom order.
        """
        if self.B is not None:
            state = self.B.index(set)

        else:
            n = len(self.values)
            # The number of elements in the target set
            num_elements = len(set)
            
            # Initialize the state
            state = 0

            # Add the number of sets with fewer elements
            for k in range(1, num_elements):
                state += len(list(combinations(range(n), k)))

            # Now, find where the target set is in the group with 'num_elements' elements
            combinations_list = list(combinations(range(n), num_elements))
            
            # Convert target_set to a sorted tuple to match the combinations
            target_tuple = tuple(sorted(set))
            
            # Find the index within the group
            idx_in_group = combinations_list.index(target_tuple)

            # Add the position within the group to the state
            state += idx_in_group

        return state
    
    def get_set(self, state):
        """
        Finds the set of basic states corresponding to a given basic/composite state.

        The sets are ordered as follows:
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1}, {0, 1, 2}, ..., {0, 1, ..., n-1}]

        Parameters:
        state (int): The state for which to find the corresponding set.

        Returns:
        set: The set corresponding to the given state.
        """

        if self.B is not None:
            return self.B[state]
        
        else:
            # the number of states
            n = len(self.values)

            # Initialize the state tracker
            current_state = 0

            # Iterate through the group sizes (1-element sets, 2-element sets, etc.)
            for k in range(1, n + 1):
                # Count the number of sets of size k
                comb_count = len(list(combinations(range(n), k)))

                # Check if the index falls within this group
                if current_state + comb_count > state:
                    # If it falls within this group, calculate the exact set
                    combinations_list = list(combinations(range(n), k))
                    set_tuple = combinations_list[state - current_state]
                    return set(set_tuple)
                
                # Otherwise, move to the next group
                current_state += comb_count

            # If the index is out of bounds, raise an error
            raise IndexError("Index out of bounds for the given set size.")


    def update_B(self, val=None):
        if val:
            self.B = val
        else:
            if len(self.values) > 0 and ((len(self.values) <= 6 and self.B_flag!='fly') or self.B_flag=='store'):
                self.B = self.gen_B()
            else:
                self.B = None


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

        try:
            return self.name == other.name
        except AttributeError:
            return self.name == other

    def __repr__(self):
        return repr(f'Variable(name={self.name}, B={self.B}, values={self.values})')



