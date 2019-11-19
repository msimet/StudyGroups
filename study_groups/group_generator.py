"""
group_generator.py: contains the main implementation of GroupGenerator, the main object for the
StudyGroups package.
"""
#pylint: disable=C0103, R0201
import itertools
import yaml
import numpy as np

class GroupGenerator:
    """
    Generates a set of small groups from a list of items (such as students in a class), attempting
    to produce a good distribution of partners/small group members for each item in the list.

    This is operated by the use of a yaml configuration file.  The file should look something like
    this:
    ```
    names:
        - Annie
        - Brian
        - Christine
        - David
        - Esther
        - Frank
    groups:
        - distribution: 2, 2, 2
          num: 4
        - distribution: 3, 3
          num: 2
    ```
    That is, we have a class with six people, and we would like to divide them into pairs 4 times
    and groups of 3 two times, for a final result of six individual groupings.

    This algorithm runs in O(n^2) time and may be unfeasible for large groups.

    Parameters:
    -----------
    yaml_file: str or path
        A YAML config file with the format above
    """
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        if 'names' not in config or 'groups' not in config:
            raise TypeError(
                "YAML file {} must have a 'names' key and a 'groups' key".format(yaml_file))
        self.n = len(config['names'])
        self.names = config['names']
        self.indices = np.arange(self.n, dtype=int)
        # matrix: the joint appearances count
        # (if student 1 & 2 appear together once, matrix[1,2]==matrix[2,1]==1).
        self.matrix = np.zeros((self.n, self.n))
        if 'seed' in config:
            np.random.seed(config['seed'])
        self.possible_combinations = {} # Hackily memoize some of the distribution calculations

    def pairs(self, divisions):
        """
            Return a set of pairs of joint appearances in a set of divisions.  For example, if the
            divisions are `[[0, 1, 4], [2, 3, 5]]`, then return a list of the pairs that appear
            together, in both permutations:
            ```
                [0, 1], [1, 0], [0, 4], [4, 0], [1, 4], [4, 1],
                [2, 3], [3, 2], [2, 5], [5, 2], [3, 5], [5, 3].
            ```
            which is useful for indexing into a numpy array.

            Parameters:
            -----------
            divisions: list of iterables
                A set of small groups whose elements collectively are the set of self.indices
            Returns:
            --------
                A list of lists as described above.
        """
        return_list = []
        for division in divisions:
            for i, element1 in enumerate(division):
                for element2 in division[i+1:]:
                    return_list.append([element1, element2])
                    return_list.append([element2, element1])
        return return_list

    def pair_matrix(self, divisions):
        """
            Return a pair matrix representing joint appearances in the subgroups defined by
            `divisions`. For example, if the divisions are `[[0, 1, 4], [2, 3, 5]]`, then return
            ```
            [[0 1 0 0 1 0]
             [1 0 0 0 1 0]
             [0 0 0 1 0 1]
             [0 0 1 0 0 1]
             [1 1 0 0 0 0]
             [0 0 1 1 0 0]]
            ```
            Note that all diagonal elements are always 0, just for convenience (we know things must
            appear with themselves).

            Parameters:
            -----------
                divisions: list of iterables
                    A set of small groups whose elements collectively are the set of self.indices
            Returns:
            --------
                pair_matrix: a np.array of size (len(self.indices), len(self.indices)) indicating
                joint appearances
        """
        matrix = np.zeros((self.n, self.n))
        pairs = self.pairs(divisions)
        pairs = ([p[0] for p in pairs], [p[1] for p in pairs])
        matrix[pairs] += 1
        return matrix

    def _generate_combinations(self, n, i=0):
        """ Produce all combinations from self.indices[i:] that have length n. """
        if n == 1:
            return [[ii] for ii in self.indices[i:]]
        return_list = []
        for ii in range(i, self.n):
            return_list.extend([[ii] + g for g in self._generate_combinations(n-1, ii+1)])
        return return_list

    def generate_combinations(self, n):
        """
            Produce all possible combinations from the list of `self.indices` with length `n`, and
            return a list of tuples where the first item is the combinations and the second is the
            matrix of indices appearing together.  For example, if `self.indices = [0, 1, 2, 3]`,
            then calling this with n=2 will return
            [((0, 1, 2), [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
             ((0, 1, 3), [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]),
             ((0, 2, 3), [[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]),
             ((1, 2, 3), [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])],
            not necessarily in that order.

            Parameters:
            -----------
                n : int
                    The number of items in the individual group
            Returns:
            --------
                return_list: a list of tuples where the first item of the tuple is a combination and
                             the second is a matrix describing which items appear together.
        """
        if n >= self.n:
            raise ValueError(
                "Requested number of items must be less than or equal to the number of 'names'")
        if n < 0:
            raise ValueError("Requested number of items must be positive")
        return self._generate_combinations(n)

    def _generate_divisions(self, dist):
        if len(dist) == 1:
            return [[pc] for pc in self.possible_combinations[dist[0]]]
        possible_divisions = []
        # Could be multiple items, so recurse.
        subsets = self._generate_divisions(dist[1:])
        for pd in self.possible_combinations[dist[0]]:
            # To avoid duplication, make sure all groups of the same length are in lexical order.
            possible_divisions.extend([[pd] + s for s in subsets
                                       if len(pd) != len(s[0]) or pd < s[0]])
        # Now get rid of possible divisions with overlapping elements.
        desired_n = sum(dist)
        possible_divisions = [pd for pd in possible_divisions
                              if len(set(itertools.chain(*pd))) == desired_n]
        return possible_divisions

    def generate_divisions(self, dist):
        """
            Generate all possible divisions of the items in self.indices into smaller divisions of
            size dist[0], dist[1], etc.  For example, if `self.indices = [0,1,2,3]`, then calling
            this function with `dist=(2,2)` would result in:
            ```
            [
                [[0, 1], [2, 3]],
                [[0, 2], [1, 3]],
                [[0, 3], [1, 2]]
            ]

            Parameters:
            -----------
                dist: tuple
                    A tuple of ints with sum(dist)==len(self.indices)
            Returns:
            --------
                return_list: a list such that each element of the list is a tuple with len(dist),
                    and each subelement of the element is one of the indices from self.indices,
                    such that set(all subelements)==set(self.indices) but no two elements contain
                    the *same* splits into groups, considering permutations of either elements or
                    subelements. len(return_list[i])==dist[i] for all i<len(dist).
        """
        if np.sum(dist) != self.n:
            raise ValueError("Sum of elements of dist must be number of elements in self.names")
        dist = sorted(dist)
        dist_set = set(dist)
        # You don't need to regenerate permutations for things you've already seen
        for ds in dist_set:
            if ds not in self.possible_combinations:
                self.possible_combinations[ds] = self.generate_combinations(ds)
        possible_divisions = self._generate_divisions(dist)
        return [(pd, self.pair_matrix(pd)) for pd in possible_divisions] # Get the pair matrix too

    def entirely_random_division(self, dist):
        """
            Return an entirely random division of self.indices into subgroups defined by dist.

            Parameters:
            -----------
                dist: list of ints
                    A list representing the desired divisions of self.indices into subgroups
            Returns:
            --------
                division: list of tuples
                    Randomly-defined subgroups of size dist
        """
        elements = np.random.permutation(self.indices)
        return_list = []
        for d in dist:
            return_list.append(tuple(sorted(elements[:d])))
            elements = elements[d:]
        return_list = sorted(return_list)
        return return_list, self.pair_matrix(return_list)

    def choose_division(self, dist):
        """
            Choose the optimal division of self.indices into groups of size dist, based on an
            accounting of which elements have been grouped together in previous calls to
            choose_division.  Essentially, choose a division such that the sum of squared elements
            of the joint appearance matrix is as small as possible.

            Parameters:
            -----------
                dist: list of ints
                    A list representing the desired divisions of self.indices into subgroups
            Returns:
            --------
                division: list of tuples
                    Subgroups of size dist that cause the sum of squared elements of (M+M_new) to
                    be as small as possible, given the current status of M, the joint appearances
                    matrix.
        """
        if np.sum(self.matrix) == 0:
            # We don't have anything yet, so just randomly split things up
            chosen_division = self.entirely_random_division(dist)
            return chosen_division
        # Find the minimum loss
        possible_divisions = self.generate_divisions(dist)
        loss = [np.sum((self.matrix + pd[1])**2) for pd in possible_divisions]
        min_loss = min(loss)
        possible_divisions = [p for p, l in zip(possible_divisions, loss) if l == min_loss]
        indx = np.random.choice(len(possible_divisions))
        return possible_divisions[indx]

    def choose_groups(self):
        # Run choose_division for every requested grouping.
        pass

    def print_groups(self):
        # Print the results as a CSV to the terminal.
        pass
