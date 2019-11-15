import numpy as np
import unittest
try:
    import study_groups as sg
except ImportError:
#    import os
#    os.path.append('..')
    import study_groups as sg

class TestInitialization(unittest.TestCase):
    """ Tests that all initialization succeeds (or fails) as intended. """
    def test_init(self):
        """ Tests that missing filenames or bad input files fail, and that correct input files are properly
            ingested.
        """
        np.testing.assert_raises(TypeError, sg.GroupGenerator)
        gg = sg.GroupGenerator('test0.yaml')
        np.testing.assert_equal(gg.n, 12)
        np.testing.assert_equal(gg.names, ['Alice', 'Bob', 'Charisma', 'Dexter', 'Emily', 'Franklin',
                                          'Greta', 'Hamlet', 'Ivy', 'Jasper', 'Katie', 'Louis'])
        np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        np.testing.assert_equal(gg.matrix, np.zeros((12, 12)))
        
        np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test1.yaml')
        np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test2.yaml')
        gg = sg.GroupGenerator('test3.yaml')
        np.testing.assert_equal(gg.n, 6)
        np.testing.assert_equal(gg.names, ['Alice', 'Bob', 'Charisma', 'Dexter', 'Emily', 'Franklin'])
        np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5])
        np.testing.assert_equal(gg.matrix, np.zeros((6, 6)))

    def test_rng_seed(self):
        """ Tests that the random number seed is actually seeding when set in a yaml file, to produce
            replicable results.
        """
        gg = sg.GroupGenerator('test0.yaml')
        ri = sg.group_generator.np.random.rand()
        np.random.seed(271)
        ri_truth = np.random.rand()
        np.testing.assert_equal(ri, ri_truth)
        
    
if __name__=='__main__':
    test_init()
