StudyGroups
===========
This module is designed for teachers who have students they would like to divide into small groups
with a good distribution of group mates (which random group organization may not do). It is run
through a yaml configuration file describing the students in the class and the requested groups.

Installation
------------

Download this code from GitHub (https://github.com/msimet/StudyGroups), then:
`pip install .`

Usage
-----

StudyGroups must be run with a yaml configuration file. These are pretty simple. The config file
must have two keys, `names` and `groups`. `names` is a list of names, while `groups` is a list of
dicts describing how to group the people in `names`. Each dict in `groups` should have two keys:
`distribution`, a list of ints describing the distribution of people (such as `[2, 2, 2]` to make 3
pairs of students out of a group of 6), and `num`, an int directing how many times to make that set
of groups (`4` in this example would make 4 sets of 3 pairs, where each set of 3 pairs divides all
6 students, and the 4 sets are designed such that no student has the same partner twice).  The
distributions must always divide all the students--you can't ask for two pairs if you have six
students.  The distributions do NOT have to be equal in number: you could ask for a group of 4 and
a group of 2 for 6 students.

The yaml file for the setup in the previous paragraph would be:
```
names:
    - Annie
    - Bianca
    - Christopher
    - David
    - Elise
    - Farley
groups:
    - distribution: [2, 2, 2]
      num: 4
```

More examples can be found in the `tests/` folder. The tests with filenames beginning
`test_failure` are designed to cause errors for unit tests to catch. Do not use those as
templates! Some of the templates also include a `seed` key, which seeds a random number generator.
We generally recommend avoiding this key except for testing purposes.

Running the code is as simple as:
```
python -m study_groups [yourfile].yaml
```

This will print a CSV file to the terminal, with the student groups in *columns*, and blank lines
between every set of subgroups. You can copy and paste it directly, or redirect the output to a
.csv file:
```
python -m study_groups week1.yaml > week1.csv
```
