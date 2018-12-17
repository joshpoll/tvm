"""
Online partial evaluator based on Type-Directed Partial Evaluation.
Currently offline, but will eventually switch to online.
"""
from tvm import relay
from tvm.relay import op
from tvm.relay.expr import ExprFunctor, SD
from typing import TypeVar, Deque, Tuple, Optional
from collections import deque
import numpy as np
from enum import Enum, auto
from tvm.relay.ir_pass import infer_type

BINARY_OPS = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "less": np.less,
    "greater": np.greater,
    "less_equal": np.less_equal,
    "greater_equal": np.greater_equal,
    "equal": np.equal,
    "not_equal": np.not_equal,
}

counter = 0

def fresh_var(string):
    var = relay.var(f"{string}{counter}")
    counter += 1
    return var

def get_np(x):
    return x.data.asnumpy()

def reify(typ: relay.Type, val: relay.Expr):
    # eta expand and update tags
    if isinstance(typ, relay.FuncType):
        # keeping it simple for now
        assert len(typ.arg_types == 1)
        x1 = fresh_var("x")
        refl_x1 = reflect(typ.arg_types[0], x1)
        static_call = relay.Call(val, refl_x1)
        reify_call = reify(typ.ret_type, static_call)
        dynamic_abs = relay.Function([x1], reify_call)
        dynamic_abs.SD = SD.DYNAMIC
        return dynamic_abs
    else:
        # TODO: this is too generous and lets through things that need to be modified
        return val

def reflect(typ: relay.Type, expr: relay.Expr):
    # eta expand and update tags (opposite of reify)
    if isinstance(typ, relay.FuncType):
        # keeping it simple for now
        assert len(typ.arg_types == 1)
        v1 = fresh_var("v")
        reify_v1 = reify(typ.arg_types[0], v1)
        dynamic_call = relay.Call(expr, reify_v1)
        dynamic_call.SD = SD.DYNAMIC
        refl_call = reflect(typ.ret_type, dynamic_call)
        static_abs = relay.Function([v1], refl_call)
        return static_abs
    else:
        # TODO: see corresponding todo in reify
        return val

# TODO: should beta reduce static terms.
# i.e. if call, lambda, and args are static then reduce? Not sure the exact
# conditions. Just make sure example can run first.
class StaticReduce(ExprFunctor):
    def __init__(self):
        super().__init__()

    def visit_constant(self, const):
        return const

    def visit_var(self, var):
        return var

    def visit_call(self, call):
        assert False

    # TODO
    def visit_global_var(self, gvar):
        assert False

    # TODO
    def visit_function(self, fn):
        assert False

    # TODO
    def visit_let(self, let):
        assert False

    # TODO: revise!!
    def visit_if(self, ite):
        assert False

    # TODO
    def visit_tuple(self, tup):
        assert False

    # TODO
    def visit_tuple_getitem(self, op):
        assert False

# i.e. partially evaluate
def residualize(expr: relay.Expr, mod: relay.Module = None):
    reified = reify(infer_type(expr, mod), expr)
    return StaticReduce().visit(reified)
