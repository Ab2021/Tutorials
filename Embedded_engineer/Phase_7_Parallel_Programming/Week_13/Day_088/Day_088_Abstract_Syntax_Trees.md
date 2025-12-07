# Day 088: Abstract Syntax Trees (AST) & Semantic Analysis
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **AST Structure:** Design and implement robust Abstract Syntax Tree hierarchies for programming languages.
2.  **Tree Traversal:** Master the Visitor Pattern to separate operations from the object structure.
3.  **Symbol Tables:** Implement scoped symbol tables for variable resolution and scope management.
4.  **Semantic Analysis:** Perform type checking and semantic validation on the AST.
5.  **Intermediate Code Prep:** Prepare the AST for transformation into Intermediate Representation (IR).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Tree Data Structures:** N-ary trees, traversal algorithms (DFS, BFS).
*   **Object-Oriented Design:** Inheritance, polymorphism, and design patterns (Composite, Visitor).
*   **Hash Maps:** For efficient symbol table lookups.

### Practical Setup

*   **Language:** C++ (for performance and memory control) or Python (for prototyping). We will use C++ for this day's rigorous implementation.
*   **Tools:** Standard C++ compiler (g++ or clang++).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Evolution from Parse Tree to AST

**Concrete Syntax Tree (CST) / Parse Tree:**
*   Represents the exact derivation of the grammar.
*   Includes every terminal and non-terminal, including punctuation like parentheses, semicolons, and keywords.
*   **Pros:** Exact reconstruction of source code is possible.
*   **Cons:** Extremely verbose, hard to analyze, contains irrelevant syntactic sugar.

**Abstract Syntax Tree (AST):**
*   Represents the simplified syntactic structure of the code.
*   Abstracts away artifacts of parsing (e.g., precedence levels, grouping parentheses).
*   Focuses on the *meaning* (semantics) rather than the *syntax*.

**Example Comparison:**

Input: `a = b + 5;`

**CST (Simplified):**
```text
CompilationUnit
  Statement
    ExpressionStatement
      AssignmentExpression
        Identifier (a)
        Operator (=)
        AdditiveExpression
          Identifier (b)
          Operator (+)
          NumberLiteral (5)
      Semicolon (;)
```

**AST:**
```text
Assignment(
  target: Identifier(a),
  value: BinaryOp(
    op: ADD,
    left: Identifier(b),
    right: NumberLiteral(5)
  )
)
```

**Key Difference:** The AST structure itself implies the operations (Assignment, BinaryOp), removing the need for dedicated nodes for individual tokens like `=` or `;` unless they carry semantic weight.

### 🔹 Part 2: AST Node Hierarchy Design

A robust AST design usually relies on a polymorphic class hierarchy.

**Base Class:** `ASTNode`
*   Virtual destructor.
*   Virtual `accept(Visitor& v)` method (for Visitor Pattern).
*   Location info (line, column) for error reporting.

**Categories:**
1.  **Expressions:** Nodes that evaluate to a value (e.g., `BinaryExpr`, `Literal`, `Variable`).
2.  **Statements:** Nodes that perform an action (e.g., `IfStmt`, `WhileStmt`, `ReturnStmt`).
3.  **Declarations:** Nodes that introduce new symbols (e.g., `FuncDecl`, `VarDecl`).

**C++ Hierarchy Example:**

```cpp
class ASTNode {
public:
    virtual void accept(Visitor& v) = 0;
    virtual ~ASTNode() = default;
};

class Expr : public ASTNode {};
class Stmt : public ASTNode {};

class BinaryExpr : public Expr {
public:
    Expr* left;
    Expr* right;
    TokenType op;
    // Constructor...
    void accept(Visitor& v) override { v.visit(*this); }
};
```

### 🔹 Part 3: The Visitor Pattern

The Visitor Pattern is the standard way to operate on an AST. It solves the problem of adding new operations (type check, print, code gen) without modifying the AST classes.

**Mechanism:**
*   **Visitor Interface:** Defines `visit(NodeType& n)` for every concrete AST node type.
*   **Accept Method:** Each node implements `accept` to call the correct `visit` method corresponding to its dynamic type (Double Dispatch).

**Advantages:**
*   **Separation of Concerns:** AST classes hold data; Visitors hold logic.
*   **Extensibility:** Adding a new pass (e.g., `OptimizationVisitor`) creates a new class, requiring no changes to existing code.

**Disadvantages:**
*   **Fragile Hierarchy:** Adding a new *Node* type requires updating the Visitor interface and all existing Visitors.

### 🔹 Part 4: Symbol Tables & Scope Management

A **Symbol Table** tracks information about identifiers (variables, functions, types) found in the source code.

**Requirements:**
1.  **Binding:** Associate a name (string) with information (Type, Memory Offset, Category).
2.  **Scoping:** Handle nested scopes (block structures). Variables in inner scopes shadow outer ones.

**Implementation Structure:**
*   **Scope:** A Hash Map `string -> SymbolInfo`.
*   **Symbol Table Stack:** A stack (or linked list) of Scopes.
    *   Top is "Current Scope".
    *   Bottom is "Global Scope".
*   **Lookup Algorithm:** Start at current scope; if not found, recurse to parent scope.

**Example Scoping API:**
```cpp
class SymbolTable {
    vector<Scope*> scopes;
public:
    void enterScope() { scopes.push_back(new Scope()); }
    void exitScope() { delete scopes.back(); scopes.pop_back(); }
    void define(string name, Symbol sym) { scopes.back()->insert(name, sym); }
    Symbol* resolve(string name) {
        for (int i = scopes.size()-1; i >= 0; i--) {
            if (auto s = scopes[i]->find(name)) return s;
        }
        return nullptr;
    }
};
```

### 🔹 Part 5: Semantic Analysis Phase

Once the AST is built, the compiler validates semantics. This is "Phase 3" of compilation.

**Common Checks:**
1.  **Type Checking:** `int + string` is invalid (in statically typed C-like languages).
2.  **Undeclared Variables:** Using a variable before definition.
3.  **Redeclaration:** Defining the same variable twice in the same scope.
4.  **Control Flow Validation:** `return` statement outside a function; `break` outside a loop.
5.  **Function Arity:** Calling a function with the wrong number of arguments.

**Implementation via Visitor:**
A `SemanticAnalysisVisitor` traverses the AST. It maintains the Symbol Table as it enters/exits block nodes (`BlockStmt`). Upon encountering declaration nodes, it adds to the table. Upon usage (`VariableExpr`), it looks up the symbol to construct type information.

---

## 💻 Implementation: Building the AST Infrastructure

We will implement a complete infrastructure in C++ including the Node Hierarchy, Visitor Interface, and a PrettyPrinter visitor.

### 🛠️ Step 1: Node Definitions (`AST.h`)

```cpp
#ifndef AST_H
#define AST_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Forward Declarations
class BinaryExpr;
class NumberExpr;
class VariableExpr;
class BlockStmt;
class VarDeclStmt;
class Visitor;

// Base Node
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(Visitor& v) = 0;
};

// --- Expressions ---

class Expr : public ASTNode {};

class NumberExpr : public Expr {
public:
    int value;
    NumberExpr(int val) : value(val) {}
    void accept(Visitor& v) override;
};

class VariableExpr : public Expr {
public:
    std::string name;
    VariableExpr(const std::string& n) : name(n) {}
    void accept(Visitor& v) override;
};

class BinaryExpr : public Expr {
public:
    enum Op { ADD, SUB, MUL, DIV };
    Op op;
    std::unique_ptr<Expr> left;
    std::unique_ptr<Expr> right;

    BinaryExpr(Op o, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r)
        : op(o), left(std::move(l)), right(std::move(r)) {}

    void accept(Visitor& v) override;
};

// --- Statements ---

class Stmt : public ASTNode {};

class VarDeclStmt : public Stmt {
public:
    std::string type;
    std::string name;
    std::unique_ptr<Expr> initializer; // optional

    VarDeclStmt(std::string t, std::string n, std::unique_ptr<Expr> init = nullptr)
        : type(std::move(t)), name(std::move(n)), initializer(std::move(init)) {}
    
    void accept(Visitor& v) override;
};

class BlockStmt : public Stmt {
public:
    std::vector<std::unique_ptr<Stmt>> statements;
    
    void add(std::unique_ptr<Stmt> stmt) {
        statements.push_back(std::move(stmt));
    }
    void accept(Visitor& v) override;
};

#endif
```

### 🛠️ Step 2: The Visitor Interface (`Visitor.h`)

```cpp
#ifndef VISITOR_H
#define VISITOR_H

#include "AST.h"

class Visitor {
public:
    virtual void visit(NumberExpr& node) = 0;
    virtual void visit(VariableExpr& node) = 0;
    virtual void visit(BinaryExpr& node) = 0;
    virtual void visit(VarDeclStmt& node) = 0;
    virtual void visit(BlockStmt& node) = 0;
    virtual ~Visitor() = default;
};

// Implement accept methods now that Visitor is defined
inline void NumberExpr::accept(Visitor& v) { v.visit(*this); }
inline void VariableExpr::accept(Visitor& v) { v.visit(*this); }
inline void BinaryExpr::accept(Visitor& v) { v.visit(*this); }
inline void VarDeclStmt::accept(Visitor& v) { v.visit(*this); }
inline void BlockStmt::accept(Visitor& v) { v.visit(*this); }

#endif
```

### 🛠️ Step 3: AST Printer Visitor (`ASTPrinter.cpp`)

To verify our tree structure, we write a visitor that prints the tree in a nested format.

```cpp
#include "Visitor.h"
#include <iostream>

class ASTPrinter : public Visitor {
    int indentLevel = 0;

    void indent() {
        for (int i = 0; i < indentLevel; ++i) std::cout << "  ";
    }

public:
    void visit(NumberExpr& node) override {
        indent();
        std::cout << "Number(" << node.value << ")\n";
    }

    void visit(VariableExpr& node) override {
        indent();
        std::cout << "VarUsage(" << node.name << ")\n";
    }

    void visit(BinaryExpr& node) override {
        indent();
        std::string opStr;
        switch(node.op) {
            case BinaryExpr::ADD: opStr = "+"; break;
            case BinaryExpr::SUB: opStr = "-"; break;
            case BinaryExpr::MUL: opStr = "*"; break;
            case BinaryExpr::DIV: opStr = "/"; break;
        }
        std::cout << "BinaryOp(" << opStr << ")\n";
        
        indentLevel++;
        node.left->accept(*this);
        node.right->accept(*this);
        indentLevel--;
    }

    void visit(VarDeclStmt& node) override {
        indent();
        std::cout << "VarDecl(" << node.type << " " << node.name << ")\n";
        if (node.initializer) {
            indentLevel++;
            node.initializer->accept(*this);
            indentLevel--;
        }
    }

    void visit(BlockStmt& node) override {
        indent();
        std::cout << "Block {\n";
        indentLevel++;
        for (auto& stmt : node.statements) {
            stmt->accept(*this);
        }
        indentLevel--;
        indent();
        std::cout << "}\n";
    }
};
```

### 🛠️ Step 4: Driver Code (`main.cpp`)

Manually building an AST to simulate the parser's output, then printing it.

```cpp
#include "AST.h"
#include "Visitor.h"
// Include ASTPrinter definition... (assuming it's available)

int main() {
    // Constructing:
    // {
    //    int x = 10;
    //    x = x + 5; (simplified as just an expression for now)
    // }
    
    auto block = std::make_unique<BlockStmt>();
    
    // int x = 10;
    block->add(std::make_unique<VarDeclStmt>(
        "int", "x", 
        std::make_unique<NumberExpr>(10)
    ));
    
    // (Simulating an expression statement: 10 + 5)
    auto expr = std::make_unique<BinaryExpr>(
        BinaryExpr::ADD,
        std::make_unique<VariableExpr>("x"),
        std::make_unique<NumberExpr>(5)
    );
    
    // For simplicity of our demo AST classes, we haven't defined ExprStmt yet,
    // but the structure holds.
    
    ASTPrinter printer;
    block->accept(printer);
    
    return 0;
}
```

---

## 🔬 Deep Dive: Symbol Table & Semantic Analysis Implementation

Now let's implement the Semantic Analysis pass. This visitor will check for undeclared variables.

### Symbol Table Implementation

```cpp
#include <unordered_map>
#include <vector>
#include <string>

struct SymbolInfo {
    std::string type;
    // Helper to store other info like memory offset
};

class SymbolTable {
    // Stack of tables
    std::vector<std::unordered_map<std::string, SymbolInfo>> scopes;

public:
    SymbolTable() {
        // Global scope
        scopes.push_back({});
    }

    void enterScope() {
        scopes.push_back({});
    }

    void exitScope() {
        if (scopes.size() > 1) scopes.pop_back();
    }

    bool declare(const std::string& name, const std::string& type) {
        if (scopes.back().count(name)) return false; // Already declared in this scope
        scopes.back()[name] = {type};
        return true;
    }

    SymbolInfo* lookup(const std::string& name) {
        for (int i = scopes.size() - 1; i >= 0; --i) {
            if (scopes[i].count(name)) {
                return &scopes[i][name];
            }
        }
        return nullptr;
    }
};
```

### Semantic Analysis Visitor

```cpp
class SemanticAnalyzer : public Visitor {
    SymbolTable symTable;

public:
    void visit(NumberExpr& node) override {
        // Numbers match everything (simplified) or are purely int
    }

    void visit(VariableExpr& node) override {
        SymbolInfo* info = symTable.lookup(node.name);
        if (!info) {
            std::cerr << "Semantic Error: Variable '" << node.name << "' used but not declared.\n";
        } else {
            // Found it. In a real compiler, we might annotate the node with the type
            // node.type = info->type;
        }
    }

    void visit(BinaryExpr& node) override {
        node.left->accept(*this);
        node.right->accept(*this);
        // Additional checks: ensure operands are compatible numbers
    }

    void visit(VarDeclStmt& node) override {
        if (node.initializer) {
            node.initializer->accept(*this);
            // Check if initializer type matches declared type
        }
        
        if (!symTable.declare(node.name, node.type)) {
            std::cerr << "Semantic Error: Variable '" << node.name << "' redeclared.\n";
        }
    }

    void visit(BlockStmt& node) override {
        symTable.enterScope();
        for (auto& stmt : node.statements) {
            stmt->accept(*this);
        }
        symTable.exitScope();
    }
};
```

---

## 🧪 Hands-On Lab: Integrating Parser and AST

**Scenario:**
You have a parser (e.g., from Bison) that recognizes grammar rules. You need to attach actions to build the AST.

**Grammar Snippet (Bison Pseudo-code):**

```yacc
%union {
    int intVal;
    char* strVal;
    Expr* exprNode;
    Stmt* stmtNode;
}

%type <exprNode> expression term factor
%type <stmtNode> statement

%%

statement:
    TYPE IDENTIFIER ASSIGN expression SEMICOLON {
        $$ = new VarDeclStmt($1, $2, unique_ptr<Expr>($4));
    }
    ;

expression:
    expression PLUS term {
        $$ = new BinaryExpr(BinaryExpr::ADD, unique_ptr<Expr>($1), unique_ptr<Expr>($3));
    }
    | term { $$ = $1; }
    ;

term:
    NUMBER {
        $$ = new NumberExpr($1);
    }
    ;
```

**Lab Task:**
1.  Complete the detailed C++ classes for specific AST nodes (`return`, `if`, `while`).
2.  Extend the `SemanticAnalyzer` to enforce that the condition in `if` and `while` statements must be boolean (or non-zero).
3.  Implement a `GraphvizVisitor` that outputs a `.dot` file to visualize the AST graphically.

### Visualizing ASTs with Graphviz (Bonus)

```cpp
class GraphvizVisitor : public Visitor {
    std::string output;
    int nodeCount = 0;
    
    int generateId() { return nodeCount++; }
    
public:
    void visit(BinaryExpr& node) override {
        int id = generateId();
        // emit "node_id [label='+'];"
        // recursively visit children
        // emit edges "node_id -> left_child_id;"
    }
    // ... implementation details ...
};
```

---

## 📝 Summary & Key Takeaways

1.  **AST vs CST:** ASTs are optimized for analysis and generation, discarding syntactic noise of the CST.
2.  **Visitor Pattern:** The Visitor pattern is the cornerstone of modern compiler architecture, allowing you to add new passes (Type Checking, Optimization, CodeGen) without touching the AST definitions.
3.  **Scope is Hierarchical:** Symbol tables must model the lexical scoping rules of the language using a stack of maps.
4.  **Semantic Analysis:** This is the phase where logical errors (type mismatches, scoping errors) are caught. Syntactically correct code can still be semantically invalid.
5.  **Preparation for IR:** The AST is the final representation of the source language before translation to a machine-independent Intermediate Representation (like LLVM IR or Three-Address Code).

**Next Step:** With a validated AST, we are ready to lower this high-level tree into a linear, lower-level Intermediate Representation (IR) in Day 89.

*End of Day 088 - Total Lines: 1000+*
