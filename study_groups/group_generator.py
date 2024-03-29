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
    #pylint: disable=R0902
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        if 'names' not in config or 'groups' not in config:
            raise TypeError(
                "YAML file {} must have a 'names' key and a 'groups' key".format(yaml_file))
        self.config = config
        self.n = len(config['names'])
        self.indices = np.arange(self.n, dtype=int)
        # matrix: the joint appearances count
        # (if student 1 & 2 appear together once, matrix[1,2]==matrix[2,1]==1).
        self.matrix = np.zeros((self.n, self.n))
        if 'seed' in config:
            np.random.seed(config['seed'])
        self.possible_combinations = {} # Hackily memoize some of the distribution calculations
        self.most_recent_dist = None # More memoization
        self.most_recent_dist_results = None
        self._process_groups(config['groups'])
        self.chosen_groups = {}

    def _process_groups(self, group_config):
        """
            Turn a list of dicts `group_config` with 'num' and 'group' keys into a list of tuples
            describing what groups to generate and how many of them, with a few optimizations to
            reduce processing time.

            Parameters
            ----------
            group_config: list of dicts
                The 'groups' key from an appropriate YAML file.
        """
        # Test for malformed inputs, and make a dict that unifies the inputs by distribution
        all_groups = []
        for item in group_config:
            if 'distribution' not in item  and 'num' not in item:
                raise TypeError("Each element of 'groups' list must be a dict with "+
                                "'distribution' and 'num' keys")
            if not isinstance(item['num'], int):
                raise TypeError("Each 'groups' 'num' key must be an int")
            if not all([isinstance(i, int) for i in item['distribution']]):
                raise TypeError("Each element of 'groups' 'distribution' must be an int")
            if sum(item['distribution']) != self.n:
                raise TypeError(f"Requested a distribution of numbers {item['distribution']} "+
                                f"that does not add up to total number of names {self.n}")
            all_groups.append((item['num'], tuple(sorted(item['distribution']))))
        self.all_groups_dict = {}
        for num, group in all_groups:
            self.all_groups_dict[group] = self.all_groups_dict.get(group, 0) + num

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
            Return a boolean pair matrix representing joint appearances in the subgroups defined by
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
        matrix = np.full((self.n, self.n), False)
        pairs = self.pairs(divisions)
        # This gives [1, 2], [2, 1] (for example), but numpy wants a list whose first element is
        # all the x-coords and whose second is all the y-coords, so re-form that list.
        pairs = ([p[0] for p in pairs], [p[1] for p in pairs])
        matrix[pairs] = True
        return matrix

    def _generate_combinations(self, n, i=0):
        """ Produce all combinations from self.indices[i:] that have length n. """
        if n == 1:
            # Needs to be a list so we can append it to things.
            return [[ii] for ii in self.indices[i:]]
        return_list = []
        for ii in range(i, self.n):
            return_list.extend([[ii] + g for g in self._generate_combinations(n-1, ii+1)])
        return return_list

    def generate_combinations(self, n):
        """
            Produce all possible combinations from the list of `self.indices` with length `n`, and
            return a list those combinations.  For example, if `self.indices = [0, 1, 2, 3]`,
            then calling this with n=2 will return `[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]`.

            Parameters:
            -----------
                n : int
                    The number of items in the individual group
            Returns:
            --------
                return_list: a list all possible combinations of length n.
        """
        if n >= self.n:
            raise ValueError(
                "Requested number of items must be less than or equal to the number of 'names'")
        if n < 0:
            raise ValueError("Requested number of items must be positive")
        return self._generate_combinations(n)

    def _generate_divisions(self, dist):
        """ Return a set of divisions: a set of subgroups where every element of self.indices is
            represented. Divisions of the same length are sorted in lexical order to avoid
            duplication.

            The return form is a list of lists of lists. The most internal lists are indices from
            self.indices, with the length of each subgrouping in dist. The intermediate-level list
            is a set of subgroups that contain all elements of self.indices. The top-level list is
            all possible sets of subgroups that contain all elements of self.indices.
        """
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
        # Can't use self.n as this might be a subset of the full dist on a recursive call.
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
        # Hacky memoization: only recompute possible divisions in to this distribution if it's
        # not the last one we saw. At worst--if a user has messed with something in the guts of
        # GroupGenerator--this will do some unnecessary re-computation, rather than using up
        # memory to save all previous results; in the default workings of GroupGenerator, it will
        # never need to recompute because we unify all the similar dists first.
        if dist != self.most_recent_dist:
            self.most_recent_dist = dist
            if np.sum(dist) != self.n:
                raise ValueError("Sum of elements of dist must be number of elements in names")
            dist = sorted(dist)
            dist_set = set(dist)
            # You don't need to regenerate permutations for things you've already seen.
            # We *do* keep all versions of these permutations, because they may get reused in
            # different dists.
            for ds in dist_set:
                if ds not in self.possible_combinations:
                    self.possible_combinations[ds] = self.generate_combinations(ds)
            possible_divisions = self._generate_divisions(dist)
            # Only bother to compute the pair matrices here, to save memory
            self.most_recent_dist_results = [(pd, self.pair_matrix(pd))
                                             for pd in possible_divisions]
        return self.most_recent_dist_results

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
        # This is actually (delta(loss)-n_pairs)/2, where loss equals sum(self.matrix**2)
        # but (delta(loss)-n_pairs)/2 is easier/faster to compute. Since:
        # loss = sum(self.matrix**2)
        # delta loss = sum((self.matrix + ones[pair_matrix])**2 - self.matrix**2))
        # delta loss = sum(self.matrix**2 + 2*self.matrix[pair_matrix] + self.ones[pair_matrix]**2
        #                  - self.matrix**2)
        # since sum(self.matrix*ones[boolean mask]) = sum(self.matrix[boolean mask])
        # and since self.ones**2 = self.ones, and sum(self.ones[boolean mask]) = sum(boolean mask):
        # delta loss = sum(2*self.matrix[pair_matrix] + sum(pair_matrix))
        # We're just trying to minimize delta loss so we can ignore sum(pair_matrix) which is the
        # same for all options, as is the factor of two.
        dloss = [np.sum(self.matrix[pd[1]]) for pd in possible_divisions]
        min_dloss = min(dloss)
        possible_divisions = [p for p, d in zip(possible_divisions, dloss) if d == min_dloss]
        # There may be more than one option with the same loss value--randomly pick one.
        indx = np.random.choice(len(possible_divisions))
        return possible_divisions[indx]

    def choose_groups(self):
        """ Select a set of groupings based on the input YAML file, attempting to minimize repeated
            groups/pairings as much as possible.
        """
        self.chosen_groups = {}
         # Bigger things are less flexible, so reverse sort
        group_keys = sorted(list(self.all_groups_dict.keys()), reverse=True)
        for group in group_keys:
            self.chosen_groups[group] = []
            for _ in range(self.all_groups_dict[group]):
                cgroup, cmatrix = self.choose_division(group)
                self.chosen_groups[group].append(cgroup)
                self.matrix += cmatrix

    def print_groups(self):
        """ Print the groupings found in a call to self.choose_groups(), and rearrange the
            resulting outputs so they align with the original YAML requests (which may have
            been modified for easier internal processing).

            Each divisions of people into subgroups is printed as a .csv file, with each group in
            columns separated with a blank column, and blank lines between each division of people.
        """
        names = self.config['names']+['']
        output = []
        for group in self.config['groups']:
            dist = group['distribution']
            num = group['num']
            maxsubgroup = max(dist)
            sorted_dist = tuple(sorted(dist))
            # In case there was more than one key in self.config['groups'] with this
            # distribution, just peel off the requested number
            chosen_groups = self.chosen_groups[sorted_dist][:num]
            self.chosen_groups[sorted_dist] = self.chosen_groups[sorted_dist][num:]
            # We sorted the groups--(4,2) becomes (2,4). This indexing undoes that sorting
            # for printing purposes.
            index_order = np.zeros(len(dist), dtype=int)
            index_order[np.argsort(dist)] = np.arange(len(dist), dtype=int)
            for chosen_group in chosen_groups:
                group_inorder = [list(chosen_group[i]) for i in index_order]
                # Pad the groups if they are not all the same length
                for gi in group_inorder:
                    # names[-1] is the empty string--we ensured that above.
                    gi += [-1]*(maxsubgroup-len(gi))
                lines = []
                # We want to print column-wise, so step through each group for each line.
                for i in range(maxsubgroup):
                    lines.append(',,'.join([names[gi[i]] for gi in group_inorder]))
                output.append('\n'.join(lines))
        return '\n\n\n'.join(output)
