from typing import List
from enum import Enum, auto
from abc import ABC

IND_COUNT = 0

class AbstractFormula(ABC):
    def __init__(self, neg=False):
        self.negated = neg

class Var(AbstractFormula):
    def __init__(self, name: str, neg=False):
        global IND_COUNT
        super().__init__(neg)
        self.index = IND_COUNT #уникальный номер переменной
        self.name = name
        IND_COUNT += 1

    def __add__(self, other):
        return Formula(Symbol.DISJUNCTION, args=[self, other])

    def __mul__(self, other):
        return Formula(Symbol.CONJUNCTION, args=[self, other])

    def __mod__(self, other):
        return Formula(Symbol.EQUIVALENCE, args=[self, other])

    def __rshift__(self, other):
        return Formula(Symbol.IMPLICATION, args=[self, other])

    def __neg__(self):
        self.negated = True
        return self

class Symbol(Enum):
    CONJUNCTION = auto()
    DISJUNCTION = auto()
    IMPLICATION = auto()
    EQUIVALENCE = auto()

class Formula(AbstractFormula ):
    def __init__(self, symbol: Symbol, args: List[AbstractFormula], neg=False):
        super().__init__(neg)
        self.symbol = symbol
        self.args = args

    def __add__(self, other):
        return Formula(Symbol.DISJUNCTION, args=[self, other])

    def __mul__(self, other):
        return Formula(Symbol.CONJUNCTION, args=[self, other])

    def __mod__(self, other):
        return Formula(Symbol.EQUIVALENCE, args=[self, other])

    def __rshift__(self, other):
        return Formula(Symbol.IMPLICATION, args=[self, other])

    def __neg__(self):
        self.negated = True
        return self

if __name__ == '__main__':
    a = Var('a')
    b = Var('b')
    c = Var('c')
    d = Var('d')
    res = (a + b) * (c >> d)
    print(12)