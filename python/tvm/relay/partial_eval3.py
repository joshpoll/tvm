"""
Online partial evaluator based on "A Type-directed, On-line, Partial Evaluator
for a Polymorphic Language" by Tim Sheard. We will eventually revisit this
choice. Currently handles only a limited subset of the language features
described in the paper.

Programs consist of static (available at compile-time) and dynamic (available at
run-time) data. The goal of a partial evaluator is to use static data to
precompute as much of the program as possible, improving runtime performance.
The partial evaluator thus subsumes constant folding and dead-code elimination.
It also simplifies complicated function applications.

This evaluator is not perfect. In some cases it may duplicate computation,
resulting in suboptimal performance. In the presence of complicated recursion,
the pass may not terminate.

For an introduction to partial evaluation see
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.9354&rep=rep1&type=pdf
- http://phoenix.inria.fr/publications/talks/introduction.pdf
- https://dl.acm.org/citation.cfm?id=243447
- https://softwarefoundations.cis.upenn.edu/plf-current/PE.html

The classic text on partial evaluation is
- https://www.itu.dk/~sestoft/pebook/jonesgomardsestoft.ps

For a description of this flavor of partial evaluation (type-directed partial
evaluation or normalization by evaluation) see
- https://pdfs.semanticscholar.org/1ef8/743d6f444f1fbad085a86ec93a5921813f2a.pdf
- Type-Directed Partial Evaluation by Danvy
- Sections 2.4, 2.5 of Normalisation by Evaluation in the Compilation of Typed
  Functional Programming Languages by Lindley

For sophisticated implementation details see
- Sections 4 and 5 of Normalisation by Evaluation in the Compilation of Typed
  Functional Programming Languages by Lindley
"""
from tvm import relay
from tvm.relay import expr
from typing import NamedTuple, Callable, List, Tuple, Union, NoReturn
from abc import ABC
from types import MethodType

class Value(ABC):
    pass

class VConstant(Value):
    value: relay.Constant

    def __init__(self, value: relay.Constant) -> None:
        self.value = value

class VFunction(Value):
    domain_type: relay.Type

    def __init__(self, domain_type: relay.Type) -> None:
        self.domain_type = domain_type

class VPair(Value):
    left: Value
    right: Value

    def __init__(self, left: Value, right: Value) -> None:
        self.left = left
        self.right = right

# coerces exprs to values
class VDyn(Value):
    expr: relay.Expr

    def __init__(self, expr: relay.Expr) -> None:
        self.expr = expr

class Eval(expr.ExprEnvFunctor2):
    def __init__(self):
        super().__init__()

    def visit_constant(self, env, const):
        return VConstant(const)

    def visit_var(self, env, var):
        expr.ExprEnvFunctor2.lookup(env, var.vid)

    # TODO: make "smart" ops
    # TODO: relax constraints
    def visit_call(self, env, call):
        assert call.attrs is None
        assert call.type_args is None
        assert len(call.args) == 1
        eval_op = eval(env, call.op)
        assert isinstance(eval_op, VFunction)
        return eval_op.func(Eval().visit(env, call.args[0]))

    # TODO: not sure how to handle this case
    def visit_global_var(self, env, gvar):
        assert False

    # TODO: relax constraints
    def visit_function(self, env, func):
        assert not func.type_params
        assert len(func.params) == 1
        f = lambda self, v: Eval().visit(expr.ExprEnvFunctor2.extend(env, func.params[0].vid, v), func.body)
        vfunc = VFunction(func.params[0].type_annotation)
        print("setting vfunc in eval")
        vfunc.func = MethodType(f, vfunc)
        return vfunc

    # TODO
    def visit_tuple(self, env, tup):
        assert False

    # TODO
    def visit_tuple_getitem(self, env, op):
        assert False

    # TODO
    def visit_let(self, _, let):
        assert False

    # TODO
    def visit_if(self, env, ite):
        assert False

def reify(v: Value):
    if isinstance(v, VConstant):
        return v.value

    if isinstance(v, VFunction):
        x = relay.var("x", type_annotation=v.domain_type)
        return relay.Function([x], reify(v.func(reflect(v.domain_type, x))))

    if isinstance(v, VDyn):
        return v.expr
    
    if isinstance(v, VPair):
        return expr.Tuple([reify(v.left), reify(v.right)])

    print(v)
    raise Exception("Unhandled case in reify!")

def reflect(t: relay.Type, e: relay.Expr):
    if isinstance(t, relay.FuncType):
        assert len(t.arg_types) == 1
        assert not t.type_params
        assert not t.type_constraints
        vfunc = VFunction(t.arg_types[0])
        print("setting vfunc in reflect")
        f = lambda self, v: reflect(t.ret_type, relay.Call(e, reify(v)))
        vfunc.func = MethodType(f, vfunc)
        return vfunc

    if isinstance(t, relay.TensorType):
        return VDyn(e)

    raise Exception("Unhandled case in reflect!")

def partial_eval(e: relay.Expr, env=None):
    if env is None:
        env = []
    return reify(Eval().visit(env, e))
