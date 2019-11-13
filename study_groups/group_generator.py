"""
group_generator.py: contains the main implementation of GroupGenerator, the main object for the StudyGroups package.
"""

class GroupGenerator:
    def __init__(self, yaml_file):
        pass
        
    
    def generate_combinations(self, n):
        # Produce a list of all possible combinations of size n from our internal set of students.
        pass
        
    def generate_divisions(self, dist):
        # Generate all possible ways to divide our internal set of students into a distribution of groups dist.
        # For example, if dist=[4,4,4] for 12 students, take the results of generate_combinations(4) and
        # produce every set of 3 combinations that contains 12 unique students.
        pass
        
    def choose_division(self, dist):
        # Given the results of generate_divisions, select the optimal group
        pass
        
    def choose_groups(self):
        # Run chose_division for every requested grouping.
        pass
        
    def print_groups(self):
        # Print the results as a CSV to the terminal.
        pass
        
