from typing import List
from enum import Enum, auto
from abc import ABC
import copy

IND_COUNT = 0


class AbstractFormula(ABC):
    def __init__(self, neg=False):
        self.negated = neg


class Var(AbstractFormula):
    def __init__(self, name: str, neg=False):
        global IND_COUNT
        super().__init__(neg)
        self.index = IND_COUNT  # уникальный номер переменной
        self.name = name
        IND_COUNT += 1

    def __add__(self, other):
        if isinstance(other, Formula) and other.symbol==Symbol.DISJUNCTION:
            other.args.append(self)
            other.sortArgs()
            return other

        return Formula(Symbol.DISJUNCTION, args=[self, other])

    def __mul__(self, other):
        if isinstance(other, Formula) and other.symbol==Symbol.CONJUNCTION:
            other.args.append(self)
            other.sortArgs()
            return other

        return Formula(Symbol.CONJUNCTION, args=[self, other])

    def __mod__(self, other):
        return Formula(Symbol.EQUIVALENCE, args=[self, other])

    def __rshift__(self, other):
        return Formula(Symbol.IMPLICATION, args=[self, other])

    def __neg__(self):
        if self.negated:
            self.negated = False
        else:
            self.negated = True
        return self

    def __str__(self):
        if self.negated:
            return '-' + self.name
        return self.name

    def __repr__(self): return self.__str__()



class Symbol(Enum):
    CONJUNCTION = '*'
    DISJUNCTION = '+'
    IMPLICATION = '=>'
    EQUIVALENCE = '<=>'


class Formula(AbstractFormula):
    def __init__(self, symbol: Symbol, args: List[AbstractFormula], neg=False):
        super().__init__(neg)
        self.symbol = symbol
        self.args = args
        self.canonize()

    def sortArgs(self):
        self.args.sort(key=lambda x: x.name if isinstance(x, Var) else 'nonsorted')

    def __add__(self, other):
        if self.symbol == Symbol.DISJUNCTION and isinstance(other, Var):
            self.args.append(other)
            self.sortArgs()
            return self

        if self.symbol == Symbol.DISJUNCTION and isinstance(other, Formula) and other.symbol == Symbol.DISJUNCTION:
            self.args.extend(other.args)
            self.sortArgs()
            return self

        return Formula(Symbol.DISJUNCTION, args=[self, other])

    def __mul__(self, other):
        if self.symbol == Symbol.CONJUNCTION and isinstance(other, Var):
            self.args.append(other)
            self.sortArgs()
            return self

        if self.symbol == Symbol.CONJUNCTION and isinstance(other, Formula) and other.symbol == Symbol.CONJUNCTION:
            self.args.extend(other.args)
            self.sortArgs()
            return self

        return Formula(Symbol.CONJUNCTION, args=[self, other])

    def __mod__(self, other):
        return Formula(Symbol.EQUIVALENCE, args=[self, other])

    def __rshift__(self, other):
        return Formula(Symbol.IMPLICATION, args=[self, other])

    def __neg__(self):
        if self.negated:
            self.negated = False
        else:
            self.negated = True
        return self

    def canonize(self):
        left = copy.deepcopy(self.args[0])
        right = copy.deepcopy(self.args[1])

        if self.symbol == Symbol.EQUIVALENCE:
            self.symbol = Symbol.CONJUNCTION
            self.args = [Formula(Symbol.IMPLICATION, args=[left, right]),
                         Formula(Symbol.IMPLICATION, args=[right, left])]

        if self.symbol == Symbol.IMPLICATION:
            self.symbol = Symbol.DISJUNCTION
            self.args = [-left, right]

        self.sortArgs()

    def __str__(self):
        if self.negated:
            res = '-('
        else:
            res = ''
        res = res + str(self.symbol.value) + '('
        for el in self.args:
            res += str(el) + ' '
        if self.negated:
            res += ')'
        return res[:len(res)-1] + ')'






if __name__ == '__main__':
    a = Var('a')
    b = Var('b')
    c = Var('c')
    d = Var('d')
    e = Var('e')
    f = Var('f')
    # res = ((a + d + b) * -c) % (d >> e >> f)
    res = -a * (f * c)
    print(res)
    print(12)
