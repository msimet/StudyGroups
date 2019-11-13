import numpy as np
try:
    import study_groups as sg
except ImportError:
#    import os
#    os.path.append('..')
    import study_groups as sg

def test_init():
    np.testing.assert_raises(TypeError, sg.GroupGenerator)
    gg = sg.GroupGenerator('test0.yaml')
    np.testing.assert_equal(gg.n, 12)
    np.testing.assert_equal(gg.names, ['Alice', 'Bob', 'Charisma', 'Dexter', 'Emily', 'Franklin',
                                      'Greta', 'Hamlet', 'Ivy', 'Jasper', 'Katie', 'Louis'])
    np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    np.testing.assert_equal(gg.matrix, np.zeros((12, 12)))
    
    # check that random number generator was seeded
    ri = sg.group_generator.np.random.rand()
    np.random.seed(271)
    ri_truth = np.random.rand()
    np.testing.assert_equal(ri, ri_truth)
    
    np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test1.yaml')
    np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test2.yaml')
    gg = sg.GroupGenerator('test3.yaml')
    np.testing.assert_equal(gg.n, 6)
    np.testing.assert_equal(gg.names, ['Alice', 'Bob', 'Charisma', 'Dexter', 'Emily', 'Franklin'])
    np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5])
    np.testing.assert_equal(gg.matrix, np.zeros((6, 6)))
    
if __name__=='__main__':
    test_init()
