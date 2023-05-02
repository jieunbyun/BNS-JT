import unittest
from io import StringIO
import importlib
import numpy as np

from BNS_JT.variable import Variable
from BNS_JT.cpm import Cpm
from BNS_JT.config import read_nodes

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test_bridge_network(unittest.TestCase):
    '''
         -x1-
       -      -
    O-         --x3--D
       -      -
         -x2-

    '''
    @classmethod
    def setUpClass(cls):

        variables = [4, 1, 2, 3]
        no_child = 1
        C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]])
        p = [1, 1, 1, 1]
        B = [[1, 0], [0, 1], [1, 1]]
        value = ['survival', 'fail']

        cls.cpms = Cpm(**{'variables': variables,
                          'no_child': no_child,
                          'C': C,
                          'p': p})
        cls.x1 = Variable(**{'B': B, 'value': value})
        cls.x2 = Variable(**{'B': B, 'value': value})
        cls.x3 = Variable(**{'B': B, 'value': value})
        cls.s = Variable(**{'B': B, 'value': value})

    def test_init(self):

        self.assertTrue(isinstance(self.cpms, Cpm))
        self.assertTrue(isinstance(self.x1, Variable))
        self.assertTrue(isinstance(self.x2, Variable))
        self.assertTrue(isinstance(self.x3, Variable))
        self.assertTrue(isinstance(self.s, Variable))


def test_read_nodes():
    file_node = StringIO("""
id,x,y
1,-2,3
2,-2,-3
3,2,-2
4,1,1
5,0,0
        """)
    node_coords = read_nodes(file_node)

    expected = {1: (-2, 3),
                2: (-2, -3),
                3: (2, -2),
                4: (1, 1),
                5: (0, 0)}

    assert node_coords == expected


if __name__=='__main__':
    unittest.main()

