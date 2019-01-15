import random

MAX_VALUE = 10000
MAX_CLASS = 100
RULES_LENGTH = 2000000
SPARSE_LEVEL = 10

def wrap(n):
    if n > MAX_VALUE:
        return "*"
    else:
        return str(n)

def random_line():
    return [ wrap(random.randint(0, SPARSE_LEVEL*MAX_VALUE)) for _ in range(10) ]
    
def random_class():
    return str(random.randint(0, MAX_CLASS))

def write_rule(fname, size_ratio=1):
    with open(fname, 'w') as rulef:
        for i in range(int(size_ratio * RULES_LENGTH)):
            rulef.write(";".join(random_line()) + ";" + random_class() + "\n")

def write_file(fname, size_ratio=1):
    with open(fname, 'w') as rulef:
        for i in range(int(size_ratio * RULES_LENGTH)):
            rulef.write(";".join(random_line()) + "\n")
        
        
if __name__ == '__main__':
    "Real input"
    ratio = 1
    write_rule("rule_2M.csv", ratio)
    for i in range(0,10):
        write_file("transactions_{}.csv".format(i), ratio)
    
    "Tiny input"
    ratio = .01
    write_rule("rule_tiny.csv", ratio)
    write_file("transactions_tiny.csv", ratio)
    