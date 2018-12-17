from tvm import relay
from tvm.relay import ExprEnvFunctor, op
from typing import TypeVar, Deque, Tuple, Optional
from collections import deque
import numpy as np

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

def get_np(x):
    return x.data.asnumpy()

# Online partial evaluator adapted from
# https://softwarefoundations.cis.upenn.edu/plf-current/PE.html
# See https://www.itu.dk/~sestoft/pebook/jonesgomardsestoft-letter.pdf,
# https://pdfs.semanticscholar.org/6656/de4e36356a999a1d02e9a6b0135a1dae081f.pdf,
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.6534&rep=rep1&type=pdf
# for additional improvements.
class PartialEvaluator(ExprEnvFunctor):
    def __init__(self):
        super().__init__()

    def visit_list(self, envs, exprs):
        return [self.visit(envs, expr) for expr in exprs]

    def visit_constant(self, envs, rconst):
        return (envs, rconst)

    def visit_var(self, envs, rvar):
        return (envs, ExprEnvFunctor.lookup(envs, rvar) or rvar)

    def visit_call(self, envs, call):
        # check if it's a binary op
        for bin_op, np_op in BINARY_OPS.items():
            if call.op == op.get(bin_op):
                assert len(call.args) == 2
                _, visited_args = zip(*self.visit_list(envs, call.args))
                
                # if all the arguments are constant, evaluate it
                if all(isinstance(arg, relay.Constant) for arg in visited_args):
                    visited_args = [get_np(arg) for arg in visited_args]
                    
                    return (envs, relay.const(np_op(visited_args[0], visited_args[1])))
                else:
                    return (envs, relay.Call(call.op, visited_args, call.attrs, call.type_args))
        else:
            raise Exception("unhandled call case!")

    # TODO
    def visit_global_var(self, envs, gvar):
        return (envs, gvar)

    # TODO
    def visit_function(self, _, fn):
          new_body = self.visit(fn.body)
          return relay.Function(
              list(fn.params),
              fn.ret_type, new_body,
              fn.type_params)

    # TODO
    def visit_let(self, _, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return relay.Let(new_var, new_val, new_body)

    # TODO: revise!!
    def visit_if(self, envs, ite):
        cond_envs, cond = self.visit(envs, ite.cond)

        if isinstance(cond, relay.Constant):
            cond_val = cond.data.asnumpy().item()
            assert isinstance(cond_val, bool)
            if cond_val:
                return self.visit(cond_envs, ite.true_branch)
            else:
                return self.visit(cond_envs, ite.false_branch)
        else:
            return (cond_envs, cond)

    # TODO
    def visit_tuple(self, tup):
        return relay.Tuple([self.visit(field) for field in tup.fields])

    # TODO
    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return relay.TupleGetItem(tuple_value, op.index)
        return op
