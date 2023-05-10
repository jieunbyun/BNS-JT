import numpy as np
from collections import namedtuple
from typing import NamedTuple


BaseVariable = namedtuple('BaseVariable', [
    'name',
    'B',
    'values'
    ])

class Variable(BaseVariable):
    '''
    A namedtuple subclass to hold Variable

    name:
    B: basis set
    values: description of states

    B may be replaced by a dictionary something like "numBasicState = 3" and "compositeState = {'4':['1','2'], '5':['1','3'], '6': ['1','2','3']}"
    (A user does not have to enter composite states for all possible permutations but is enough define those being used).
    '''
    __slots__ = ()
    def __new__(cls, name, B, values):

        assert isinstance(B, (np.ndarray, list)), 'B must be a array'

        if isinstance(B, list):
            B = np.array(B)

        assert isinstance(values, list), 'values must be a list'

        numBasicState = B.shape[1]

        assert (B[:numBasicState, :] == np.eye(numBasicState)).all(), 'The upper part corresponding to basic states must form an identity matrix'

        return super(Variable, cls).__new__(cls, name, B, values)

    def __hash__(self):

        return hash(self.name)

    def __eq__(self, other):

        return self.name == other.name

    def __repr__(self):
        return repr(f'Variable(name={self.name}, B={self.B}, values={self.values})')

"""

    def __init__(self, name, B, values):

        assert isinstance(name, str), 'name should be a string'

        self.name = name

        assert isinstance(B, (np.ndarray, list)), 'B must be a array'

        if isinstance(B, list):
            self.B = np.array(B)
        else:
            self.B = B

        assert isinstance(values, (np.ndarray, list)), 'values must be a vector'

        if isinstance(values, list):
            self.values = np.array(values)
        else:
            self.values = values

        numBasicState = self.B.shape[1]

        assert (self.B[:numBasicState, :] == np.eye(numBasicState)).all(), 'The upper part corresponding to basic states must form an identity matrix'

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return NotImpemented
        return self.name == other.name and np.array_equal(self.B, other.B) and np.array_equal(self.values, other.values)

    def __repr__(self):
        return repr(f'Variable(B={self.B}, values={self.values})')
"""
