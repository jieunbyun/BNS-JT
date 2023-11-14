import numpy as np
from collections import namedtuple
from typing import NamedTuple

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

    def __init__(self, name, B, values):

        assert isinstance(name, str), 'name should be a string'

        assert isinstance(B, (np.ndarray, list)), 'B must be a array'

        if isinstance(B, list):
            B = np.array(B, dtype=int)

        if B.dtype == np.dtype(np.float64):
            B = B.astype(int)

        assert isinstance(values, list), 'values must be a list'

        num_basicstate = B.shape[1]

        assert (B[:num_basicstate, :] == np.eye(num_basicstate)).all(), 'The upper part corresponding to basic states must form an identity matrix'

        self.name = name
        self.B = B
        self.values = values

    def B_times_values(self):

        return [' '.join(x).strip(' ') for x in np.char.multiply(self.values, self.B.astype(int)).tolist()]

    def __hash__(self):

        return hash(self.name)

    def __lt__(self, other):

        return self.name < other.name

    def __eq__(self, other):

        return self.name == other.name

    def __repr__(self):
        return repr(f'Variable(name={self.name}, B={self.B}, values={self.values})')

