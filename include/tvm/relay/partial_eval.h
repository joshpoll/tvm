/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/interpreter.h
 * \brief An interpreter for Relay.
 *
 * This file implements a simple reference interpreter for Relay programs.
 * Given a Relay module, and a Relay expression it produces a value.
 *
 * The interpreter's values are a naive representation of the values that
 * can be produced by a Relay program and are exposed via tvm::Node's
 * system to Python for introspection and debugging.
 *
 * The interpreter's intent is to serve as a reference semantics for the Relay IR,
 * as well as for debugging and testing.
 */
#ifndef TVM_RELAY_INTERPRETER_H_
#define TVM_RELAY_INTERPRETER_H_

#include <tvm/build_module.h>
#include <tvm/relay/module.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
namespace partial_eval {

/*!
 * \brief A Relay value.
 */
class Value;

/*! \brief The base container type of Relay values. */
class ValueNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Value";
  TVM_DECLARE_BASE_NODE_INFO(ValueNode, RelayNode);
};

class Value : public NodeRef {
 public:
  Value() {}
  explicit Value(NodePtr<Node> n) : NodeRef(n) {}
  const ValueNode* operator->() const {
    return static_cast<const ValueNode*>(node_.get());
  }

  using ContainerType = ValueNode;
};

class VTensor;

class VTensorNode : public ValueNode {
  public:
    Constant val;

    VTensorNode() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("val", &val);
    }

    TVM_DLL static VTensor make(Constant val);

    static constexpr const char* _type_key = "relay.VTensor";
    TVM_DECLARE_NODE_TYPE_INFO(VTensorNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(VTensor, VTensorNode, Value);

class VFun;

class VFunNode : public ValueNode {
  public:
    FuncType type;
    std::function<Value(tvm::Array<Value>)> func;

    VFunNode() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("type", &type);
      // can't export func
    }

    TVM_DLL static VFun make(FuncType type, std::function<Value(tvm::Array<Value>)> func);

    static constexpr const char* _type_key = "relay.VFun";
    TVM_DECLARE_NODE_TYPE_INFO(VFunNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(VFun, VFunNode, Value);

class VOp;

class VOpNode : public ValueNode {
  public:
    Op op;

    VOpNode() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("op", &op);
    }

    TVM_DLL static VOp make(Op op);

    static constexpr const char* _type_key = "relay.VOp";
    TVM_DECLARE_NODE_TYPE_INFO(VOpNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(VOp, VOpNode, Value);

class VTuple;

class VTupleNode : public ValueNode {
  public:
    tvm::Array<Value> fields;

    VTupleNode() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("fields", &fields);
    }

    TVM_DLL static VTuple make(tvm::Array<Value> fields);

    static constexpr const char* _type_key = "relay.VTuple";
    TVM_DECLARE_NODE_TYPE_INFO(VTupleNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(VTuple, VTupleNode, Value);

class VDyn;

class VDynNode : public ValueNode {
  public:
    Expr expr;

    VDynNode() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("expr", &expr);
    }

    TVM_DLL static VDyn make(Expr expr);

    static constexpr const char* _type_key = "relay.VDyn";
    TVM_DECLARE_NODE_TYPE_INFO(VDynNode, ValueNode);
};

RELAY_DEFINE_NODE_REF(VDyn, VDynNode, Value);

struct Env {
  // vars and their evaluated values
  std::map<Var, Value> vars;

  // static lookup
  static Value Lookup(const Var& var, const Env& env) {
    auto val = env.vars.find(var);
    if (val == env.vars.end()) {
      LOG(FATAL) << "Couldn't find `" << var << "` in environment.";
    }
    return env.vars.at(var);
  }

  // static extend
  static Env Extend(const Var& var, const Value& value, const Env& env) {
    Env new_env = env;
    new_env.vars.insert({var, value});
    return new_env;
  }
};

// A simple Relay interpreter. Differs from existing interpreter, because
// function values have a different type and this interpreter passes around an
// immutable environment.
// Also implements Op behavior differently. Instead of separating dumb and smart
// functions by a tag as in the original online TDPE paper, we treat all
// Functions as dumb and all Ops as smart.
class Interp : public ExprFunctor<Value(const Expr& n, const Env& env)> {
  public:
    Value Eval(const Expr& expr, const Env& env);
    Value VisitExpr(const Expr& expr, const Env& env);
    Value VisitExpr_(const ConstantNode* const_node, const Env& env);
    Value VisitExpr_(const VarNode* var_node, const Env& env);
    Value VisitExpr_(const LetNode* let_node, const Env& env);
    Value VisitExpr_(const FunctionNode* func_node, const Env& env);
    Value VisitExpr_(const OpNode* op_node, const Env& env);
    Value VisitExpr_(const CallNode* call_node, const Env& env);
    Value VisitExpr_(const TupleNode* tuple_node, const Env& env);
    Value VisitExpr_(const TupleGetItemNode* op, const Env& env);
};

Expr Reify(const Value& value);
Value Reflect(const Type& type, const Expr& expr);
Expr PartialEval(const Expr& expr, const Env& env);

}  // namespace partial_eval
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_INTERPRETER_H_
