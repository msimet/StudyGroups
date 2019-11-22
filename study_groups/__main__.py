import sys
from .group_generator import GroupGenerator

obj = GroupGenerator(sys.argv[1])
obj.choose_groups()
print(obj.print_groups())
