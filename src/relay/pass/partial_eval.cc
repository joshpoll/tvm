/*!
 * Copyright (c) 2018 by Josh Pollock
 * \file partial_eval.cc
 *
 * An online partial evaluator based on "A Type-directed, On-line, Partial
 * Evaluator for a Polymorphic Language" by Tim Sheard. We will eventually
 * revisit this choice. Currently handles only a limited subset of the language
 * features described in the paper.
 *
 * Programs consist of static (available at compile-time) and dynamic (available
 * at run-time) data. The goal of a partial evaluator is to use static data to
 * precompute as much of the program as possible, improving runtime performance.
 * The partial evaluator thus subsumes constant folding and dead-code
 * elimination. It also simplifies complicated function applications.
 *
 * This evaluator is not perfect. In some cases it may duplicate computation,
 * resulting in suboptimal performance. In the presence of complicated
 * recursion, the pass may not terminate.
 *
 * For an introduction to partial evaluation see
 * - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.9354&rep=rep1&
 *   * type=pdf
 * - http://phoenix.inria.fr/publications/talks/introduction.pdf
 * - https://dl.acm.org/citation.cfm?id=243447
 * - https://softwarefoundations.cis.upenn.edu/plf-current/PE.html
 *
 * The classic text on partial evaluation is
 * - https://www.itu.dk/~sestoft/pebook/jonesgomardsestoft.ps
 *
 * For a description of this flavor of partial evaluation (type-directed partial
 * evaluation or normalization by evaluation) see
 * - https://pdfs.semanticscholar.org/1ef8/743d6f444f1fbad085a86ec93a5921813f2a.pdf
 * - Type-Directed Partial Evaluation by Danvy
 * - Sections 2.4, 2.5 of Normalisation by Evaluation in the Compilation of
 *   Typed Functional Programming Languages by Lindley
 *
 * For sophisticated implementation details see
 * - Sections 4 and 5 of Normalisation by Evaluation in the Compilation of Typed
 *   Functional Programming Languages by Lindley
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/partial_eval.h>
#include <tvm/relay/pass.h>
#include <tvm/packed_func_ext.h>
#include "../backend/compile_engine.h"

namespace tvm {
namespace relay {
namespace partial_eval {

/* Value Implementation */
VTensor VTensorNode::make(Constant val) {
  NodePtr<VTensorNode> n = make_node<VTensorNode>();
  n->val = val;
  return VTensor(n);
}

TVM_REGISTER_API("relay._make.VTensor")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VTensorNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VTensorNode>([](const VTensorNode* node, tvm::IRPrinter* p) {
    p->stream << "VTensorNode(" << node->val << ")";
  });

VFun VFunNode::make(FuncType type, std::function<Value(tvm::Array<Value>)> func) {
  NodePtr<VFunNode> n = make_node<VFunNode>();
  n->type = type;
  n->func = func;
  return VFun(n);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VFunNode>([](const VFunNode* node, tvm::IRPrinter* p) {
    p->stream << "VFunNode(" << "TODO" << ")";
  });

VOp VOpNode::make(Op op) {
  NodePtr<VOpNode> n = make_node<VOpNode>();
  n->op = op;
  return VOp(n);
}

TVM_REGISTER_API("relay._make.VOp")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VOpNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VOpNode>([](const VOpNode* node, tvm::IRPrinter* p) {
    p->stream << "VOpNode(" << node->op << ")";
  });

VTuple VTupleNode::make(tvm::Array<Value> fields) {
  NodePtr<VTupleNode> n = make_node<VTupleNode>();
  n->fields = fields;
  return VTuple(n);
}

TVM_REGISTER_API("relay._make.VTuple")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VTupleNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VTupleNode>([](const VTupleNode* node, tvm::IRPrinter* p) {
    p->stream << "VTupleNode(" << node->fields << ")";
  });

VDyn VDynNode::make(Expr expr) {
  NodePtr<VDynNode> n = make_node<VDynNode>();
  n->expr = std::move(expr);
  return VDyn(n);
}

TVM_REGISTER_API("relay._make.VDyn")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VDynNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VDynNode>([](const VDynNode* node, tvm::IRPrinter* p) {
    p->stream << "VDynNode(" << node->expr << ")";
  });

/* Interpreter */

Value InvokePrimitiveOp(Function func,
                          const Array<Value>& args) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;

  auto engine_ = CompileEngine::Global();

  Target target_ = Target::create("llvm");
  // Marshal the arguments.
  // Handle tuple input/output by flattening them.
  size_t arg_len = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].as<VTensorNode>()) {
      ++arg_len;
    } else {
      const auto* tvalue = args[i].as<VTupleNode>();
      arg_len += tvalue->fields.size();
    }
  }
  size_t num_inputs = arg_len;
  if (const auto* tuple_type = func->body->checked_type().as<TupleTypeNode>()) {
    arg_len += tuple_type->fields.size();
  } else {
    CHECK(func->body->checked_type().as<TensorTypeNode>());
    arg_len += 1;
  }
  std::vector<TVMValue> values(arg_len);
  std::vector<int> codes(arg_len);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  auto fset_input = [&](size_t i, Value val) {
    const VTensorNode* tv = val.as<VTensorNode>();
    CHECK(tv != nullptr) << "expect Tensor argument";
    setter(i, tv->val->data);
    DLContext arg_ctx = tv->val->data->ctx;
    CHECK(arg_ctx.device_type ==  ctx.device_type &&
          arg_ctx.device_id == ctx.device_id);
      /* << "Interpreter expect context to be "
      << ctx << ", but get " << arg_ctx; */
  };

  int arg_counter = 0;
  for (Value arg : args) {
    if (arg.as<VTensorNode>()) {
      fset_input(arg_counter++,  arg);
    } else {
      const VTupleNode* tuple = arg.as<VTupleNode>();
      CHECK(tuple != nullptr);
      for (size_t i = 0; i < tuple->fields.size(); ++i) {
        fset_input(arg_counter++, tuple->fields[i]);
      }
    }
  }

  // TVM's calling convention is that the final argument is the output
  // buffer. To preserve the illusion of being a functional language
  // we need to allocate space for the output buffer based on the
  // return type.
  auto fset_output = [&](size_t i, Type val_type) {
    const TensorTypeNode* rtype = val_type.as<TensorTypeNode>();
    CHECK(rtype != nullptr);
    // Allocate output tensor.
    std::vector<int64_t> shape;
    for (auto dim : rtype->shape) {
      const auto* ivalue = as_const_int(dim);
      CHECK(ivalue) << "expected concrete dimensions";
      shape.push_back(ivalue[0]);
    }
    DLDataType dtype = Type2TVMType(rtype->dtype);
    auto out_tensor = VTensorNode::make(
        ConstantNode::make(runtime::NDArray::Empty(shape, dtype, ctx)));
    setter(num_inputs + i, out_tensor->val->data);
    return out_tensor;
  };

  PackedFunc packed_func = engine_->JIT(CCacheKeyNode::make(func, target_));
  TVMRetValue rv;
  if (const TupleTypeNode* rtype = func->body->checked_type().as<TupleTypeNode>()) {
    Array<Value> fields;
    for (size_t i = 0; i < rtype->fields.size(); ++i) {
      fields.push_back(fset_output(i, rtype->fields[i]));
    }
    packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
    return VTupleNode::make(fields);
  } else {
    Value out_tensor = fset_output(0, func->body->checked_type());
    packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
    return out_tensor;
  }
}

// Check if function is a primitive function.
bool IsPrimitive(const Function& func) {
  NodeRef res = FunctionGetAttr(func, "Primitive");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

// run the interpreter
Value Interp::Eval(const Expr& expr, const Env& env) {
  return (*this)(expr, env);
}

Value Interp::VisitExpr(const Expr& expr, const Env& env) {
  auto ret = ExprFunctor<Value(const Expr& n, const Env& env)>::VisitExpr(expr, env);
  return ret;
}

Value Interp::VisitExpr_(const ConstantNode* const_node, const Env& env) {
  return VTensorNode::make(GetRef<Constant>(const_node));
}

Value Interp::VisitExpr_(const VarNode* var_node, const Env& env) {
  auto var = GetRef<Var>(var_node);
  return Env::Lookup(var, env);
}

Value Interp::VisitExpr_(const LetNode* let_node, const Env& env) {
  auto evaled_value = VisitExpr(let_node->value, env);
  return VisitExpr(let_node->body, Env::Extend(let_node->var, evaled_value, env));
}

Value Interp::VisitExpr_(const FunctionNode* func_node, const Env& env) {
  auto func = GetRef<Function>(func_node);
  std::function<Value(tvm::Array<Value>)> value_func;
  // If func is a primitive op (i.e. has been generated from a lowered op), the
  // function value should call InvokePrimitiveOp. Otherwise, the function value
  // should evaluate the body of the function with an extended environment that
  // contains the input parameters.
  if (IsPrimitive(func)) {
    value_func = [=](tvm::Array<Value> inputs) -> Value {
      return InvokePrimitiveOp(func, inputs);
    };
  } else {
    value_func = [=](tvm::Array<Value> inputs) -> Value {
      Env extended_env = env;
      for (int i = 0; i < func_node->params.size(); i++) {
        extended_env = Env::Extend(func_node->params[i], inputs[i], extended_env);
      }
      return VisitExpr(func_node->body, extended_env);
    };
  }

  tvm::Array<Type> domain_type;
  for (auto param : func_node->params) {
    domain_type.push_back(param->type_annotation);
  }

  return VFunNode::make(FuncTypeNode::make(domain_type, func_node->ret_type, {}, {}), value_func);
}

Value Interp::VisitExpr_(const OpNode* op_node, const Env& env) {
  return VOpNode::make(GetRef<Op>(op_node));
}

// Treat ops as "smart" functions (cf. TDPE).
// Notice that it is impossible to apply e.g. add to a purely dynamic node. It
// must always be applied to a "tuple". Not sure if this affects semantics in
// any significant way. Possible remedy is adding a "smart" call node like the
// hybrid online-offline partial eval paper.
// TODO: 7.1 and 7.2
Value Interp::VisitExpr_(const CallNode* call_node, const Env& env) {
  tvm::Array<Value> args;
  for (auto arg : call_node->args) {
    args.push_back(Eval(arg, env));
  }

  // TODO: probably want to reify arguments in every case

  Value fn_val = Eval(call_node->op, env);
  if (auto vop = fn_val.as<VOpNode>()) {
    bool exists_dynamic_value = false;
    for (auto arg : args) {
      if (const VDynNode* dyn_node = arg.as<VDynNode>()) {
        exists_dynamic_value = true;
        break;
      }
    }

    if (exists_dynamic_value) {
      // reify the args
      tvm::Array<Expr> reified_args;
      for (auto arg : args) {
        reified_args.push_back(Reify(arg));
      }
      return VDynNode::make(CallNode::make(call_node->op, reified_args));
    } else {
      // all arguments are static, so we can evaluate it.
      Expr prepped_call = InferType(GetRef<Call>(call_node), {});
      prepped_call = FuseOps(prepped_call, 0);
      prepped_call = InferType(prepped_call, {});
      return VisitExpr(prepped_call, env);
    }
  } else if (const VFunNode* fun_node = fn_val.as<VFunNode>()) {
    return fun_node->func(args);
  } else {
    LOG(FATAL) << "internal error: type error, expected function value in the call "
                << "position";
    return Value();
  }
}

Value Interp::VisitExpr_(const TupleNode* tuple_node, const Env& env) {
  tvm::Array<Value> evaluated_fields;
  for (auto field : tuple_node->fields) {
    evaluated_fields.push_back(Eval(field, env));
  }
  return VTupleNode::make(evaluated_fields);
}

Value Interp::VisitExpr_(const TupleGetItemNode* op, const Env& env) {
  Value val = Eval(op->tuple, env);
  auto tuple_node = val.as<VTupleNode>();
  CHECK(tuple_node)
    << "internal error: when evaluating TupleGetItem expected a tuple value";
  CHECK_LT(static_cast<size_t>(op->index), tuple_node->fields.size())
      << "internal error: index out of bounds";
  return tuple_node->fields[op->index];
}

// Reify (with the help of Reflect) turns a Value back into an Expr
Expr Reify(const Value& value) {
  if (const VTensorNode* constant = value.as<VTensorNode>()) {
    return constant->val;
  } else if (const VDynNode* dynamic = value.as<VDynNode>()) {
    return dynamic->expr;
  } else if (const VFunNode* function = value.as<VFunNode>()) {
    // clone params
    tvm::Array<Var> eta_vars;
    for (auto type : function->type->arg_types) {
      eta_vars.push_back(VarNode::make("x", type));
    }

    // reflect vars
    tvm::Array<Value> reflected_vars;
    for (auto var : eta_vars) {
      reflected_vars.push_back(Reflect(var->type_annotation, var));
    }

    return FunctionNode::make(eta_vars, Reify(function->func(reflected_vars)), function->type->ret_type, {});
  } else if (const VTupleNode* tuple = value.as<VTupleNode>()) {
  // | VPair(a, b) => EPair(reify(a), reify(b))
    tvm::Array<Expr> reified_fields;
    for (auto field : tuple->fields) {
      reified_fields.push_back(Reify(field));
    }
    return TupleNode::make(reified_fields);
  } else { assert(false); }
}

Value Reflect(const Type& type, const Expr& expr) {
  if (const TensorTypeNode* tensor_type = type.as<TensorTypeNode>()) {
    return VDynNode::make(expr);
  } else if (const FuncTypeNode* func_type = type.as<FuncTypeNode>()) {
    auto func = [=](tvm::Array<Value> input_args) -> Value {
      tvm::Array<Expr> reified_args;
      for (auto arg : input_args) {
        reified_args.push_back(Reify(arg));
      }
      return Reflect(func_type->ret_type, CallNode::make(expr, reified_args));
    };
    return VFunNode::make(GetRef<FuncType>(func_type), func);
  } else if (const TupleTypeNode* tuple_type = type.as<TupleTypeNode>()) {
    tvm::Array<Value> reflected_fields;
    for (int i = 0; i < tuple_type->fields.size(); i++) {
      auto new_expr = TupleGetItemNode::make(GetRef<Tuple>(expr.as<TupleNode>()), i);
      reflected_fields.push_back(Reflect(tuple_type->fields[i], new_expr));
    }
    return VTupleNode::make(reflected_fields);
  } else { assert(false); }
}

Expr PartialEval(const Expr& expr, const Env& env) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  Target target = Target::create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  BuildConfigContext fresh_build_ctx(build_config());

  return Reify(Interp().Eval(expr, env));
};

TVM_REGISTER_API("relay._ir_pass.PartialEval")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = PartialEval(args[0], {});
});

}  // namespace partial_eval
}  // namespace relay
}  // namespace tvm