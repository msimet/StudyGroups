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

class TestMatrices(unittest.TestCase):
    """ Tests that joint appearance matrices can be created from input lists of indices. """
    def setUp(self):
        """ Initialize a group generator to test against """
        self.gg = sg.GroupGenerator('test3.yaml')

    def test_pairs(self):
        """ Check that all joint-appearance pairs are correctly obtained """
        input1 = [[0, 1], [2, 3], [4, 5]]
        output1 = [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4]]
        result1 = self.gg.pairs(input1)
        np.testing.assert_equal(result1, output1)

        input2 = [[0, 1, 2], [3, 4, 5]]
        output2 = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                   [3, 4], [4, 3], [3, 5], [5, 3], [4, 5], [5, 4]]
        result2 = self.gg.pairs(input2)
        np.testing.assert_equal(result2, output2)

        input3 = ((0, 1, 2, 3, 4, 5),) # make sure this works with other iterables
        output3 = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0], [0, 5], [5, 0],
                   [1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [4, 1], [1, 5], [5, 1], [2, 3], [3, 2],
                   [2, 4], [4, 2], [2, 5], [5, 2], [3, 4], [4, 3], [3, 5], [5, 3], [4, 5], [5, 4]]
        result3 = self.gg.pairs(input3)
        np.testing.assert_equal(result3, output3)

        input4 = [[0], [1, 2], [3, 4, 5]]
        output4 = [[1, 2], [2, 1], [3, 4], [4, 3], [3, 5], [5, 3], [4, 5], [5, 4]]
        result4 = self.gg.pairs(input4)
        np.testing.assert_equal(result4, output4)

    def test_pair_matrix(self):
        """ Test that joint appearances are correctly registered in a pair matrix """
        input1 = [[0, 1], [2, 3], [4, 5]]
        output1 = np.array([[0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0]])
        result1 = self.gg.pair_matrix(input1)
        np.testing.assert_equal(result1, output1)

        input2 = [[0, 1, 2], [3, 4, 5]]
        output2 = np.array([[0, 1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 1, 0]])
        result2 = self.gg.pair_matrix(input2)
        np.testing.assert_equal(result2, output2)

        input3 = ((0, 1, 2, 3, 4, 5),) # make sure this works with other iterables
        output3 = np.array([[0, 1, 1, 1, 1, 1],
                            [1, 0, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1],
                            [1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 0]])
        result3 = self.gg.pair_matrix(input3)
        np.testing.assert_equal(result3, output3)

        input4 = [[0], [1, 2], [3, 4, 5]]
        output4 = np.array([[0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 1, 0]])
        result4 = self.gg.pair_matrix(input4)
        np.testing.assert_equal(result4, output4)

class TestCombinations(unittest.TestCase):
    """ Tests that combinations of indices and corresponding pair matrices can be
        correctly created.
    """
    def setUp(self):
        """ Initialize a group generator to test against """
        self.gg = sg.GroupGenerator('test3.yaml')

    def add_pair_matrix(self, ret):
        """ Add a pair matrix for generate_combinations comparisons """
        return [(r, self.gg.pair_matrix([r])) for r in ret]

    def test_internal_combinations(self):
        """ Test the internal code that generates combinations of indices """
        results = self.gg._generate_combinations(1, i=5)
        output = [[5]]
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(1, i=3)
        output = [[3], [4], [5]]
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(2, i=3)
        output = [[3, 4], [3, 5], [4, 5]]
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(2, i=0)
        output = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                  [1, 2], [1, 3], [1, 4], [1, 5], [2, 3],
                  [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(4, i=3)
        output = []
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(3, i=1)
        output = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
                  [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        np.testing.assert_equal(results, output)

        results = self.gg._generate_combinations(6)
        output = [[0, 1, 2, 3, 4, 5]]
        np.testing.assert_equal(results, output)

    def test_combinations(self):
        """ Test that the full set of combinations is generated for each length """
        np.testing.assert_raises(ValueError, self.gg.generate_combinations, 7)

        results = self.gg.generate_combinations(2)
        output = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                  [1, 2], [1, 3], [1, 4], [1, 5], [2, 3],
                  [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
        np.testing.assert_equal(results, self.add_pair_matrix(output))

        results = self.gg.generate_combinations(3)
        output = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],
                  [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],
                  [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
                  [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        np.testing.assert_equal(results, self.add_pair_matrix(output))

        results = self.gg.generate_combinations(1)
        output = [[0], [1], [2], [3], [4], [5]]
        np.testing.assert_equal(results, self.add_pair_matrix(output))

        results = self.gg.generate_combinations(0)
        output = []
        np.testing.assert_equal(results, self.add_pair_matrix(output))

if __name__ == '__main__':
    unittest.main()
