import random

MAX_VALUE = 10000
MAX_CLASS = 100
RULES_LENGTH = 2000000
TR_LENGTH = 1000000
TR_FILES = 2
SPARSE_LEVEL = 10

def wrap(n):
    if n >= MAX_VALUE:
        return "*"
    else:
        return str(n)

def random_line(sparse = SPARSE_LEVEL):
    line = [ wrap(random.randint(0, sparse*MAX_VALUE)) for _ in range(9)]
    line.insert(random.randint(0,10),str(random.randint(0,MAX_VALUE)))
    return line
    
def random_class():
    return str(random.randint(0, MAX_CLASS))

def write_rule(fname, size_ratio=1):
    with open(fname, 'w') as rulef:
        for i in range(int(size_ratio * RULES_LENGTH)):
            rulef.write(";".join(random_line()) + ";" + random_class() + "\n")

def write_file(fname, size_ratio=1):
    with open(fname, 'w') as rulef:
        for i in range(int(size_ratio * TR_LENGTH)):
            rulef.write(";".join(random_line(sparse = 1)) + "\n")
        
        
if __name__ == '__main__':
    "Real input"
    ratio = 1
    write_rule("rule_2M.csv", ratio)
    for i in range(0,TR_FILES):
        write_file("transactions_{}.csv".format(i), ratio)
    
    "Tiny input"
    ratio = .01
    write_rule("rule_tiny.csv", ratio)
    write_file("transactions_tiny.csv", ratio)
