/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/doc.h
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#ifndef TVM_RELAY_IR_DOC_H_
#define TVM_RELAY_IR_DOC_H_

#include <tvm/relay/expr.h>
#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

// ADT
struct DocAtomNode {
  virtual ~DocAtomNode() = default;
};

using DocAtom = std::shared_ptr<DocAtomNode>;

struct TextNode : DocAtomNode {
  std::string str;

  TextNode(const std::string& str) : str(str) {}
};

struct LineNode : DocAtomNode {
  int indent;

  LineNode(int indent) : indent(indent) {}
};

// Doc is a stream-like interface
struct Doc {
 public:
  Doc() {}
  Doc(const DocAtom& atom) : stream_({atom}) {}

  // Append right to left.
  Doc& operator<<(const Doc& right);
  // like above, but automatically lifts string to a doc atom
  Doc& operator<<(const std::string& right);
  // like above, but converts right to a string first
  template<typename T>
  Doc& operator<<(const T& right) {
    std::ostringstream os;
    os << right;
    return *this << os.str();
  }

  // indent a doc stream
  friend Doc Indent(int indent, const Doc& doc);

  std::string str() {
    std::ostringstream os;
    for (auto atom : stream_) {
      if (auto text = std::dynamic_pointer_cast<TextNode>(atom)) {
        os << text->str;
      } else if (auto line = std::dynamic_pointer_cast<LineNode>(atom)) {
        os << "\n" << std::string(line->indent, ' ');
      } else {assert(false);}
    }
    return os.str();
  }
 private:
  std::vector<DocAtom> stream_;
};

// DSL functions

// text constructor
DocAtom Text(const std::string& str);
// line constructor
DocAtom Line(int indent = 0);

// render vectors of docs with a separator. e.g. [1, 2, 3], f -> 1f2f3
Doc PrintVec(const std::vector<Doc>& vec, const Doc& sep = Text(", "));
// Print constant bool value.
Doc PrintBool(bool value);
Doc PrintDType(DataType dtype);
Doc PrintString(const std::string& value);
/*!
 * \brief special method to print out const scalar
 * \param dtype The data type
 * \param data The pointer to hold the data.
 */
template<typename T>
Doc PrintConstScalar(DataType dtype, const T* data) {
  std::ostringstream os;
  if (dtype == Int(32)) {
    os << data[0];
  } else if (dtype == Float(32)) {
    os << data[0] << 'f';
  } else if (dtype == Bool()) {
      return PrintBool(data[0] != 0);
  } else {
    os << dtype << "(" << data[0] << ")";
  }
  return Text(os.str());
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_IR_DOC_H_
