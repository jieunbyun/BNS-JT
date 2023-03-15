import numpy as np

class Variable(object):
    '''

    B: basis set
    value: description of states

    B may be replaced by a dictionary something like "numBasicState = 3" and "compositeState = {'4':['1','2'], '5':['1','3'], '6': ['1','2','3']}"
    (A user does not have to enter composite states for all possible permutations but is enough define those being used).

    '''

    def __init__(self, B, value):

        assert isinstance(B, (np.ndarray, list)), 'B must be a array'

        if isinstance(B, list):
            self.B = np.array(B)
        else:
            self.B = B

        assert isinstance(value, (np.ndarray, list)), 'value must be a vector'
        if isinstance(value, list):
            self.value = np.array(value)
        else:
            self.value = value

        numBasicState = self.B.shape[1]

        assert (self.B[:numBasicState, :] == np.eye(numBasicState)).all(), 'The upper part corresponding to basic states must form an identity matrix'

    def __repr__(self):
        return repr(f'Variable(B={self.B}, value={self.value})')

