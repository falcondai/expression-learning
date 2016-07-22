from functools import *
from numpy import random
import sys

class Expression:
    def eval(self):
        raise NotImplemented()

class Function(Expression):
    def __init__(self, function_name, f, ary, *args):
        self.function_name = function_name
        self.f = f
        self.ary = ary
        self.args = args

    def __str__(self):
        args_str = ','.join(map(str, self.args))
        return '%s(%s)' % (self.function_name, args_str)

    def eval(self):
        args_eval = map(lambda x: x.eval(), self.args)
        return self.f(*args_eval)

class Constant(Expression):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{0:b}'.format(self.value)

    def __repr__(self):
        return self.value

    def eval(self):
        return self.value

class Symbol(Expression):
    def __init__(self, literal):
        self.literal = literal

    def __str__(self):
        return str(self.literal)

    def eval(self):
        return self

def random_exp(ops, constants, p_exp):
    # choose an expression uniformly randomly
    op = random.choice(ops)
    args = []
    for i in xrange(2):
        # choose to build another random expression with probability `p_exp`
        if random.rand() < p_exp:
            exp = random_exp(ops, constants, p_exp)
        else:
            exp = random.choice(constants)
        args.append(exp)
    return op(*args)

def main():
    plus = partial(Function, '+', lambda x, y: x+y, 2)
    times = partial(Function, 'x', lambda x, y: x*y, 2)
    power = partial(Function, '^', lambda x, y: x**y, 2)
    minus = partial(Function, '-', lambda x, y: x-y, 2)
    ops = [plus, ]

    zero = Constant(0)
    one = Constant(1)
    two = Constant(2)
    a = Symbol('a')

    # print plus(one, a)
    # e1 = times(two, plus(one, two))
    # e2 = power(two, zero)
    # print e1, e1.eval()
    # print e2, e2.eval()

    random.seed(int(sys.argv[4]))

    const = []
    for i in xrange(int(sys.argv[1])):
        const.append(Constant(random.randint(0, 2**20)))
    # print const

    for i in xrange(int(sys.argv[2])):
        exp = random_exp(ops, const, float(sys.argv[3]))
        print exp, '{0:b}'.format(exp.eval())

if __name__ == '__main__':
    main()
