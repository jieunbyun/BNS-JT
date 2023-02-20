import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test1(unittest.TestCase):

    #@classmethod
    #def setUpClass(cls):

        #cls.dmg_by_agent = get_dmg_by_agent()
        #cls.thresholds_by_agent = get_thresholds_by_agent()

    def test_init(self):

        variables = [1, 2, 3]
        numChild = 1
        C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
        p = [1, 1, 1]

        S_givenX1andX2 = Cpm(**{'variables': variables,
                                'numChild': numChild,
                                'C': C,
                                'p': p})



if __name__=='__main__':
    unittest.main()

