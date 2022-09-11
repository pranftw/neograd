from .autograd import tensor
from .autograd import add, sub, mul, div, _pow, exp, log, dot, _sum, transpose
from .autograd.graph import Graph


global _NG_GRAPH
_NG_GRAPH = Graph()