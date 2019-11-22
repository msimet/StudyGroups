""" test_group_generator: test the GroupGenerator() object of StudyGroups. """
#pylint: disable=C0103,R0201,W0212

import unittest
import numpy as np
try:
    import study_groups as sg
except ImportError:
    import sys
    sys.path.append('..')
    import study_groups as sg

class TestInitialization(unittest.TestCase):
    """ Tests that all initialization succeeds (or fails) as intended. """
    def test_init(self):
        """ Tests that missing filenames or bad input files fail, and that correct input files
            are properly ingested.
        """
        np.testing.assert_raises(TypeError, sg.GroupGenerator)
        gg = sg.GroupGenerator('test0.yaml')
        np.testing.assert_equal(gg.n, 12)
        np.testing.assert_equal(gg.config['names'], ['Alice', 'Bob', 'Charisma', 'Dexter',
                                                     'Emily', 'Franklin', 'Greta', 'Hamlet',
                                                     'Ivy', 'Jasper', 'Katie', 'Louis'])
        np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        np.testing.assert_equal(gg.matrix, np.zeros((12, 12)))

        np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test_failure1.yaml')
        np.testing.assert_raises(TypeError, sg.GroupGenerator, 'test_failure2.yaml')
        gg = sg.GroupGenerator('test3.yaml')
        np.testing.assert_equal(gg.n, 6)
        np.testing.assert_equal(gg.config['names'], ['Alice', 'Bob', 'Charisma',
                                                     'Dexter', 'Emily', 'Franklin'])
        np.testing.assert_equal(gg.indices, [0, 1, 2, 3, 4, 5])
        np.testing.assert_equal(gg.matrix, np.zeros((6, 6)))

    def test_rng_seed(self):
        """ Tests that the random number seed is actually seeding when set in a yaml file, to
            produce replicable results.
        """
        #pylint: disable=W0612
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
        np.testing.assert_equal(results, output)

        results = self.gg.generate_combinations(3)
        output = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],
                  [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],
                  [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
                  [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        np.testing.assert_equal(results, output)

        results = self.gg.generate_combinations(1)
        output = [[0], [1], [2], [3], [4], [5]]
        np.testing.assert_equal(results, output)

        results = self.gg.generate_combinations(0)
        output = []
        np.testing.assert_equal(results, output)

class TestDivisions(unittest.TestCase):
    """ Tests that indices can be correctly partitioned into groups of different sizes.
    """
    def make_output_pair(self, gg, ret):
        """ Generate a pair matrix to compare to the situations where that's an output """
        return [(r, gg.pair_matrix(r)) for r in ret]

    def test_internal_divisions(self):
        """ Check that unique and correctly-ordered divisions can be generated """
        gg = sg.GroupGenerator('test3.yaml')
        gg.possible_combinations[2] = gg.generate_combinations(2)
        gg.possible_combinations[1] = gg.generate_combinations(1)
        gg.possible_combinations[4] = gg.generate_combinations(4)
        results = gg._generate_divisions([2])
        np.testing.assert_equal(results, [[gc] for gc in gg.generate_combinations(2)])

        results = gg._generate_divisions([2, 2])
        output = [[[0, 1], [2, 3]],
                  [[0, 1], [2, 4]],
                  [[0, 1], [2, 5]],
                  [[0, 1], [3, 4]],
                  [[0, 1], [3, 5]],
                  [[0, 1], [4, 5]],
                  [[0, 2], [1, 3]],
                  [[0, 2], [1, 4]],
                  [[0, 2], [1, 5]],
                  [[0, 2], [3, 4]],
                  [[0, 2], [3, 5]],
                  [[0, 2], [4, 5]],
                  [[0, 3], [1, 2]],
                  [[0, 3], [1, 4]],
                  [[0, 3], [1, 5]],
                  [[0, 3], [2, 4]],
                  [[0, 3], [2, 5]],
                  [[0, 3], [4, 5]],
                  [[0, 4], [1, 2]],
                  [[0, 4], [1, 3]],
                  [[0, 4], [1, 5]],
                  [[0, 4], [2, 3]],
                  [[0, 4], [2, 5]],
                  [[0, 4], [3, 5]],
                  [[0, 5], [1, 2]],
                  [[0, 5], [1, 3]],
                  [[0, 5], [1, 4]],
                  [[0, 5], [2, 3]],
                  [[0, 5], [2, 4]],
                  [[0, 5], [3, 4]],
                  [[1, 2], [3, 4]],
                  [[1, 2], [3, 5]],
                  [[1, 2], [4, 5]],
                  [[1, 3], [2, 4]],
                  [[1, 3], [2, 5]],
                  [[1, 3], [4, 5]],
                  [[1, 4], [2, 3]],
                  [[1, 4], [2, 5]],
                  [[1, 4], [3, 5]],
                  [[1, 5], [2, 3]],
                  [[1, 5], [2, 4]],
                  [[1, 5], [3, 4]],
                  [[2, 3], [4, 5]],
                  [[2, 4], [3, 5]],
                  [[2, 5], [3, 4]]]
        np.testing.assert_equal(results, output)

        results = gg._generate_divisions([2, 2, 2])
        output = [[[0, 1], [2, 3], [4, 5]],
                  [[0, 1], [2, 4], [3, 5]],
                  [[0, 1], [2, 5], [3, 4]],
                  [[0, 2], [1, 3], [4, 5]],
                  [[0, 2], [1, 4], [3, 5]],
                  [[0, 2], [1, 5], [3, 4]],
                  [[0, 3], [1, 2], [4, 5]],
                  [[0, 3], [1, 4], [2, 5]],
                  [[0, 3], [1, 5], [2, 4]],
                  [[0, 4], [1, 2], [3, 5]],
                  [[0, 4], [1, 3], [2, 5]],
                  [[0, 4], [1, 5], [2, 3]],
                  [[0, 5], [1, 2], [3, 4]],
                  [[0, 5], [1, 3], [2, 4]],
                  [[0, 5], [1, 4], [2, 3]]]
        np.testing.assert_equal(results, output)

        results = gg._generate_divisions([4, 2])
        output = [[[0, 1, 2, 3], [4, 5]],
                  [[0, 1, 2, 4], [3, 5]],
                  [[0, 1, 2, 5], [3, 4]],
                  [[0, 1, 3, 4], [2, 5]],
                  [[0, 1, 3, 5], [2, 4]],
                  [[0, 1, 4, 5], [2, 3]],
                  [[0, 2, 3, 4], [1, 5]],
                  [[0, 2, 3, 5], [1, 4]],
                  [[0, 2, 4, 5], [1, 3]],
                  [[0, 3, 4, 5], [1, 2]],
                  [[1, 2, 3, 4], [0, 5]],
                  [[1, 2, 3, 5], [0, 4]],
                  [[1, 2, 4, 5], [0, 3]],
                  [[1, 3, 4, 5], [0, 2]],
                  [[2, 3, 4, 5], [0, 1]]]
        np.testing.assert_equal(results, output)

        results = gg._generate_divisions([1, 1, 1, 1, 1, 1])
        output = [[[0], [1], [2], [3], [4], [5]]]
        np.testing.assert_equal(results, output)

    def test_divisions(self):
        """ Test that the whole process of dividing *all* indices into groups, uniquely, is
            successful
        """
        gg = sg.GroupGenerator('test3.yaml')
        np.testing.assert_raises(ValueError, gg.generate_divisions, [2])

        results = gg.generate_divisions([2, 2, 2])
        output = [[[0, 1], [2, 3], [4, 5]],
                  [[0, 1], [2, 4], [3, 5]],
                  [[0, 1], [2, 5], [3, 4]],
                  [[0, 2], [1, 3], [4, 5]],
                  [[0, 2], [1, 4], [3, 5]],
                  [[0, 2], [1, 5], [3, 4]],
                  [[0, 3], [1, 2], [4, 5]],
                  [[0, 3], [1, 4], [2, 5]],
                  [[0, 3], [1, 5], [2, 4]],
                  [[0, 4], [1, 2], [3, 5]],
                  [[0, 4], [1, 3], [2, 5]],
                  [[0, 4], [1, 5], [2, 3]],
                  [[0, 5], [1, 2], [3, 4]],
                  [[0, 5], [1, 3], [2, 4]],
                  [[0, 5], [1, 4], [2, 3]]]
        np.testing.assert_equal(results, self.make_output_pair(gg, output))

        results = gg.generate_divisions([4, 2])
        output = [[[0, 1], [2, 3, 4, 5]],
                  [[0, 2], [1, 3, 4, 5]],
                  [[0, 3], [1, 2, 4, 5]],
                  [[0, 4], [1, 2, 3, 5]],
                  [[0, 5], [1, 2, 3, 4]],
                  [[1, 2], [0, 3, 4, 5]],
                  [[1, 3], [0, 2, 4, 5]],
                  [[1, 4], [0, 2, 3, 5]],
                  [[1, 5], [0, 2, 3, 4]],
                  [[2, 3], [0, 1, 4, 5]],
                  [[2, 4], [0, 1, 3, 5]],
                  [[2, 5], [0, 1, 3, 4]],
                  [[3, 4], [0, 1, 2, 5]],
                  [[3, 5], [0, 1, 2, 4]],
                  [[4, 5], [0, 1, 2, 3]]]
        np.testing.assert_equal(results, self.make_output_pair(gg, output))

        np.testing.assert_equal(gg.generate_divisions([4, 2]), gg.generate_divisions([2, 4]))

        results = gg.generate_divisions([1, 1, 1, 1, 1, 1])
        output = [[[0], [1], [2], [3], [4], [5]]]
        np.testing.assert_equal(results, self.make_output_pair(gg, output))

class TestSelections(unittest.TestCase):
    """ Test that optimal groups are selected given a joint appearances matrix
    """
    def test_random_selection(self):
        """ Test the random selection (when self.matrix==0) """
        gg = sg.GroupGenerator('test3.yaml')
        results = gg.choose_division([2, 2, 2])[0]
        np.random.seed(217)
        output = np.random.permutation(gg.indices).reshape(3, 2)
        output = sorted([tuple(sorted(o)) for o in output])
        np.testing.assert_equal(results, output)

    def test_choices(self):
        """ Test that good groups are chosen when some groups have already been made """
        gg = sg.GroupGenerator('test3.yaml')
        gg.matrix = np.array([[0, 0, 1, 1, 1, 1],
                              [0, 0, 1, 1, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0]])
        results = gg.choose_division([2, 2, 2])[0] # [0] to nip off the pair matrix
        output = [[0, 1], [2, 3], [4, 5]]
        np.testing.assert_equal(results, output)

        gg.matrix = np.array([[0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0]])
        # Do this a bunch of times to ensure that the first selection wasn't randomly correct
        for _ in range(30):
            result = gg.choose_division([2, 2, 2])[0]
            output_options = [[[0, 1], [2, 3], [4, 5]],
                              [[0, 1], [2, 5], [3, 4]],
                              [[0, 3], [1, 2], [4, 5]],
                              [[0, 3], [1, 5], [2, 4]],
                              [[0, 4], [1, 3], [2, 5]],
                              [[0, 4], [1, 5], [2, 3]],
                              [[0, 5], [1, 2], [3, 4]],
                              [[0, 5], [1, 3], [2, 4]]]
            self.assertIn(result, output_options)

        for _ in range(30):
            result = gg.choose_division([4, 2])[0]
            output_options = [[[4, 5], [0, 1, 2, 3]],
                              [[3, 4], [0, 1, 2, 5]],
                              [[2, 5], [0, 1, 3, 4]],
                              [[2, 4], [0, 1, 3, 5]],
                              [[2, 3], [0, 1, 4, 5]],
                              [[1, 5], [0, 2, 3, 4]],
                              [[1, 3], [0, 2, 4, 5]],
                              [[1, 2], [0, 3, 4, 5]],
                              [[0, 5], [1, 2, 3, 4]],
                              [[0, 4], [1, 2, 3, 5]],
                              [[0, 3], [1, 2, 4, 5]],
                              [[0, 1], [2, 3, 4, 5]]]
            self.assertIn(result, output_options)

class TestMultiAssignment(unittest.TestCase):
    """ Test that reasonable groups are selected for a given yaml file
    """
    def test_six(self):
        """ Test the yaml file with six names """
        gg = sg.GroupGenerator('test3.yaml')
        gg.choose_groups()
        np.testing.assert_equal(sorted(list(gg.chosen_groups.keys())), [(2, 2, 2), (2, 4), (3, 3)])
        np.testing.assert_equal(len(gg.chosen_groups[(2, 2, 2)]), 15)
        np.testing.assert_equal(len(gg.chosen_groups[(2, 4)]), 15)
        np.testing.assert_equal(len(gg.chosen_groups[(3, 3)]), 10)
        # You would think that doing this would result in the set of all possible combinations;
        # in practice, groups are occasionally repeated even though that isn't the optimal solution
        # (the solutions found here are merely "very good"). So we may not get exactly 14 overlaps
        # which is what we would get for the perfect solution--within 1 is okay though.
        self.assertTrue(np.all(np.abs(gg.matrix - 14*np.ones((6, 6)) + 14*np.diag(np.ones(6)))
                               <= np.ones((6, 6))))

    def test_twelve(self):
        """ Test the yaml file with twelve names """
        gg = sg.GroupGenerator('test0.yaml')
        gg.choose_groups()
        np.testing.assert_equal(sorted(list(gg.chosen_groups.keys())),
                                [(2, 2, 2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4), (6, 6)])
        np.testing.assert_equal(len(gg.chosen_groups[(2, 2, 2, 2, 2, 2)]), 22)
        np.testing.assert_equal(len(gg.chosen_groups[(3, 3, 3, 3)]), 22)
        np.testing.assert_equal(len(gg.chosen_groups[(4, 4, 4)]), 22)
        np.testing.assert_equal(len(gg.chosen_groups[(6, 6)]), 22)
        self.assertTrue(np.all(np.abs(gg.matrix - 22*np.ones((12, 12)) + 22*np.diag(np.ones(12)))
                               <= np.ones((12, 12))))


class TestPrinting(unittest.TestCase):
    """ Test that given a set of groups, the proper output is printed. """
    def test_printer(self):
        """ Test the print_groups() function of GroupGenerator. """
        gg = sg.GroupGenerator('test4.yaml')
        gg.chosen_groups = {
            (2, 2, 2): [[(0, 1), (2, 3), (4, 5)],
                        [(0, 2), (1, 3), (4, 5)],
                        [(0, 3), (1, 4), (2, 5)],
                        [(0, 4), (2, 3), (1, 5)],
                        [(0, 5), (1, 2), (3, 4)],
                        [(0, 1), (2, 5), (3, 4)]],
            (2, 4):    [[(0, 1), (2, 3, 4, 5)],
                        [(2, 3), (0, 1, 4, 5)]],
            (3, 3):    [[(0, 1, 2), (3, 4, 5)],
                        [(0, 2, 4), (1, 3, 5)],
                        [(0, 3, 5), (1, 2, 4)]]
        }
        expected_results = ("Alice,,Charisma,,Emily\n"+
                            "Bob,,Dexter,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Bob,,Emily\n"+
                            "Charisma,,Dexter,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Bob,,Charisma\n"+
                            "Dexter,,Emily,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Dexter\n"+
                            "Bob,,Emily\n"+
                            "Charisma,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Bob\n"+
                            "Charisma,,Dexter\n"+
                            "Emily,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Bob\n"+
                            "Dexter,,Charisma\n"+
                            "Franklin,,Emily\n"+
                            "\n\n"+
                            "Alice,,Charisma,,Bob\n"+
                            "Emily,,Dexter,,Franklin\n"+
                            "\n\n"+
                            "Alice,,Bob,,Dexter\n"+
                            "Franklin,,Charisma,,Emily\n"+
                            "\n\n"+
                            "Alice,,Charisma,,Dexter\n"+
                            "Bob,,Franklin,,Emily\n"+
                            "\n\n"+
                            "Charisma,,Alice\n"+
                            "Dexter,,Bob\n"+
                            "Emily,,\n"+
                            "Franklin,,\n"+
                            "\n\n"+
                            "Charisma,,Alice\n"+
                            "Dexter,,Bob\n"+
                            ",,Emily\n"+
                            ",,Franklin")
        results = gg.print_groups()
        np.testing.assert_equal(results, expected_results)

if __name__ == '__main__':
    unittest.main()
