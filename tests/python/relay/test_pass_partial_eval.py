import numpy as np
import tvm
from tvm import relay
# from tvm.relay.partial_eval import PartialEvaluator
# from tvm.relay.partial_eval3 import partial_eval
from tvm.relay.ir_pass import alpha_equal, partial_eval, infer_type
from collections import deque
from tvm.relay.parser import fromtext
from nose.tools import nottest

def partial_eval_equiv(before, expected):
    processed_before = infer_type(fromtext(before))
    processed_expected = infer_type(fromtext(expected))
    partial_evaled = partial_eval(processed_before)
    print(partial_evaled)
    return alpha_equal(partial_evaled, processed_expected)

def test_id():
    assert partial_eval_equiv(
        "fn (%x: int32) { %x }",
        "fn (%x: int32) { %x }"
    )

# types are messed up. may want to wait for unification to be fixed and for
# polymorphism support in partial evaluator
@nottest
def test_skk():
    assert partial_eval_equiv(
#         let s_comb_exp = EAbs(TInt, "f", EAbs(TInt, "g", EAbs(TInt, "x", EApp(EApp(EVar("f"), EVar("x")),
# EApp(EVar("g"), EVar("x"))))));
# let k_comb_exp = EAbs(TInt, "a", EAbs(TInt, "b", EVar("a")));
# let skk_exp = EApp(EApp(s_comb_exp, k_comb_exp), k_comb_exp);
        """
            let %S =
                fn (%f) {
                fn (%g: fn (int32) -> int32) {
                fn (%x: int32) {
                    %f(%x)(%g(%x))
                }
                }
                };
            let %K =
                fn (%a: int32) {
                fn (%b: int32) {
                    %a
                }
                };
            %S(%K)(%K)
        """,
        "fn (%x: int32) { %x }"
    )

def test_dce_simple():
    assert partial_eval_equiv(
        "let %x = 2; 3",
        "3"
    )

def test_dce_in_function():
    assert partial_eval_equiv(
        """
        fn (%y: int32) {
            let %x = 2;
            %y
        }
        """,
        "fn (%x: int32) { %x }"
    )

def test_tuple_simple():
    assert partial_eval_equiv(
        "(let %x = 2; 3, (fn (%x: int32) {%x})(1))",
        "(3, 1)"
    )

def test_tuple_projection():
    assert partial_eval_equiv(
        "(fn (%x: int32) {%x}, let %x = 2; 3).0",
        "fn (%x: int32) {%x}"
    )

def test_bin_ops():
    assert partial_eval_equiv(
        "1 + 2",
        "3"
    )

    assert partial_eval_equiv(
        "1 - 2",
        "-1"
    )

    assert partial_eval_equiv(
        "1 * 2",
        "2"
    )

    # assert partial_eval_equiv(
    #     "1 / 2",
    #     "0.5"
    # )

    assert partial_eval_equiv(
        "1 < 2",
        "True"
    )

    assert partial_eval_equiv(
        "1 > 2",
        "False"
    )

    assert partial_eval_equiv(
        "1 <= 2",
        "True"
    )

    assert partial_eval_equiv(
        "1 >= 2",
        "False"
    )

    assert partial_eval_equiv(
        "1 == 2",
        "False"
    )

    assert partial_eval_equiv(
        "1 != 2",
        "True"
    )

    # evaluates under lambda
    assert partial_eval_equiv(
        "fn (%x: int32) { 1 + 2 }",
        "fn (%x: int32) { 3 }"
    )

    # unless one or more of its arguments is symbolic
    assert partial_eval_equiv(
        "fn (%x: int32) { %x + 1 }",
        "fn (%x: int32) { %x + 1 }"
    )

    # but it can do some compound expressions
    assert partial_eval_equiv(
        "fn (%x: int32) { 1 + 2 + %x }",
        "fn (%x: int32) { 3 + %x }"
    )

    # but not all (due to order of operations)
    # (%x + 1) + 2 can't reduce but (1 + 2) + %x can.
    assert partial_eval_equiv(
        "fn (%x: int32) { %x + 1 + 2 }",
        "fn (%x: int32) { %x + 1 + 2 }"
    )

    assert partial_eval_equiv(
        "fn (%x: int32, %y: int32) { %x + %y }",
        "fn (%x: int32, %y: int32) { %x + %y }"
    )

    assert partial_eval_equiv(
        "(fn (%x: int32, %y: int32) { %x + %y })(2, 3)",
        "5"
    )

# def test_fold_if():
#     assert partial_eval_equiv(
#         relay.If(
#             relay.const(True),
#             relay.const(0),
#             relay.const(1)
#         ),
#         relay.const(0)
#     )

#     assert partial_eval_equiv(
#         relay.If(
#             relay.const(False),
#             relay.const(0),
#             relay.const(1)
#         ),
#         relay.const(1)
#     )

# def test_fold_const():
#     c_data = np.array([1, 2, 3]).astype("float32")
#     def before():
#         c = relay.const(c_data)
#         x = relay.var("x")
#         y = relay.add(c, c)
#         y = relay.multiply(y, relay.const(2, "float32"))
#         y = relay.add(x, y)
#         z = relay.add(y, c)
#         return relay.Function([x], z)

#     def expected():
#         x = relay.var("x")
#         c_folded = (c_data + c_data) * 2
#         y = relay.add(x, relay.const(c_folded))
#         z = relay.add(y, relay.const(c_data))
#         return relay.Function([x], z)

#     def fail(x):
#         raise RuntimeError()
#     # the fold constant should work on any context.
#     with tvm.build_config(add_lower_pass=[(0, fail)]):
#         with tvm.target.create("cuda"):
#             zz = relay.ir_pass.fold_constant(before())
#     zexpected = expected()
#     assert relay.ir_pass.alpha_equal(zz, zexpected)


# def test_fold_let():
#     c_data = np.array(1).astype("float32")
#     def before():
#         sb = relay.ScopeBuilder()
#         x = relay.var("x")
#         t1 = sb.let("t1", relay.const(c_data))
#         t2 = sb.let("t2", relay.add(t1, t1))
#         t3 = sb.let("t3", relay.add(t2, x))
#         sb.ret(t3)
#         return relay.Function([x], sb.get())

#     def expected():
#         sb = relay.ScopeBuilder()
#         x = relay.var("x")
#         c_folded = (c_data + c_data)
#         t3 = sb.let("t3", relay.add(relay.const(c_folded), x))
#         sb.ret(t3)
#         return relay.Function([x], sb.get())

#     zz = relay.ir_pass.fold_constant(before())
#     zexpected = expected()
#     assert relay.ir_pass.graph_equal(zz, zexpected)


# def test_fold_tuple():
#     c_data = np.array(1).astype("float32")
#     def before():
#         c = relay.const(c_data)
#         x = relay.var("x")
#         y = relay.Tuple([x, c])
#         z = relay.add(y[1], c)
#         z = relay.add(z, y[0])
#         return relay.Function([x], z)

#     def expected():
#         c = relay.const(c_data + c_data)
#         x = relay.var("x")
#         z = relay.add(c, x)
#         return relay.Function([x], z)

#     zz = relay.ir_pass.fold_constant(before())
#     zexpected = expected()
#     assert relay.ir_pass.graph_equal(zz, zexpected)


# def test_fold_concat():
#     c_data = np.array([[1, 2, 3]]).astype("float32")

#     def before():
#         a = relay.const(c_data)
#         b = relay.const(c_data)
#         y = relay.concatenate((a, b), axis=0)
#         return relay.Function([], y)

#     def expected():
#         y_data = np.concatenate((c_data, c_data), axis=0)
#         y = relay.const(y_data)
#         return relay.Function([], y)

#     zz = relay.ir_pass.fold_constant(before())
#     zexpected = expected()
#     assert relay.ir_pass.graph_equal(zz, zexpected)
