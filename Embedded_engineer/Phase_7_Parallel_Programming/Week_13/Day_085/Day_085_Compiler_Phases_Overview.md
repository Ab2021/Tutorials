# Day 085: Compiler Phases Overview & Architecture
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Compiler Pipeline:** Understand the complete compilation process from source to executable.
2.  **Lexical Analysis:** Tokenize source code using finite automata and regular expressions.
3.  **Syntax Analysis:** Parse token streams into Abstract Syntax Trees (AST) using context-free grammars.
4.  **Semantic Analysis:** Perform type checking, symbol resolution, and semantic validation.
5.  **Intermediate Representation:** Generate platform-independent IR for optimization and code generation.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Formal Languages:** Regular expressions, context-free grammars, finite automata.
*   **Data Structures:** Trees, graphs, hash tables, symbol tables.
*   **Programming Languages:** Understanding of language syntax and semantics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Compilation Pipeline

**Traditional Compiler Phases:**
```
Source Code
    ↓
[1] Lexical Analysis (Scanner)
    ↓
Token Stream
    ↓
[2] Syntax Analysis (Parser)
    ↓
Abstract Syntax Tree (AST)
    ↓
[3] Semantic Analysis
    ↓
Annotated AST
    ↓
[4] Intermediate Code Generation
    ↓
Intermediate Representation (IR)
    ↓
[5] Optimization
    ↓
Optimized IR
    ↓
[6] Code Generation
    ↓
Assembly/Machine Code
    ↓
[7] Linking
    ↓
Executable
```

**Modern Compiler Architecture (LLVM/GCC):**
*   **Frontend:** Language-specific (C/C++/Rust) → IR
*   **Middle-end:** IR optimization (platform-independent)
*   **Backend:** IR → machine code (platform-specific)

**Advantages of Multi-Phase Design:**
1.  **Modularity:** Each phase has well-defined input/output.
2.  **Reusability:** Share middle-end across multiple frontends/backends.
3.  **Optimization:** Apply transformations at appropriate abstraction level.
4.  **Debugging:** Isolate errors to specific phases.

### 🔹 Part 2: Lexical Analysis (Scanning)

**Purpose:**
Convert character stream into token stream.

**Token Types:**
*   **Keywords:** `if`, `while`, `return`, `int`
*   **Identifiers:** Variable/function names
*   **Literals:** Numbers, strings, characters
*   **Operators:** `+`, `-`, `*`, `/`, `=`, `==`
*   **Delimiters:** `(`, `)`, `{`, `}`, `;`, `,`

**Example:**
```c
int x = 42 + y;
```

**Token Stream:**
```
[KEYWORD, "int"]
[IDENTIFIER, "x"]
[OPERATOR, "="]
[NUMBER, "42"]
[OPERATOR, "+"]
[IDENTIFIER, "y"]
[DELIMITER, ";"]
```

**Implementation Techniques:**

1.  **Hand-Written Scanner:**
```cpp
enum TokenType { KEYWORD, IDENTIFIER, NUMBER, OPERATOR, DELIMITER };

struct Token {
    TokenType type;
    std::string lexeme;
    int line, column;
};

class Lexer {
    std::string input;
    size_t pos = 0;
    
public:
    Token nextToken() {
        skipWhitespace();
        
        if (isalpha(input[pos])) {
            return scanIdentifierOrKeyword();
        } else if (isdigit(input[pos])) {
            return scanNumber();
        } else if (isOperator(input[pos])) {
            return scanOperator();
        }
        // ... handle other cases
    }
};
```

2.  **Lexer Generator (Flex/Lex):**
```lex
%{
#include "parser.h"
%}

%%
"int"       { return KEYWORD_INT; }
"if"        { return KEYWORD_IF; }
[a-zA-Z_][a-zA-Z0-9_]*  { yylval.str = strdup(yytext); return IDENTIFIER; }
[0-9]+      { yylval.num = atoi(yytext); return NUMBER; }
"+"         { return PLUS; }
"-"         { return MINUS; }
%%
```

**Finite Automata:**
Lexical analysis is based on **Deterministic Finite Automata (DFA)**.

**Example (Identifier Recognition):**
```
State 0 (start): 
    [a-zA-Z_] → State 1
    
State 1 (accepting):
    [a-zA-Z0-9_] → State 1
    other → accept and return
```

**Performance:**
*   **Time Complexity:** $O(n)$ where $n$ is input length.
*   **Space Complexity:** $O(1)$ for DFA state machine.

### 🔹 Part 3: Syntax Analysis (Parsing)

**Purpose:**
Verify grammatical structure and build Abstract Syntax Tree (AST).

**Context-Free Grammar (CFG):**
```
Expression → Expression + Term
           | Expression - Term
           | Term

Term → Term * Factor
     | Term / Factor
     | Factor

Factor → NUMBER
       | IDENTIFIER
       | ( Expression )
```

**Parsing Techniques:**

1.  **Top-Down Parsing (Recursive Descent):**
```cpp
class Parser {
    std::vector<Token> tokens;
    size_t pos = 0;
    
    ASTNode* parseExpression() {
        ASTNode* left = parseTerm();
        
        while (match(PLUS) || match(MINUS)) {
            Token op = previous();
            ASTNode* right = parseTerm();
            left = new BinaryOp(op, left, right);
        }
        
        return left;
    }
    
    ASTNode* parseTerm() {
        ASTNode* left = parseFactor();
        
        while (match(STAR) || match(SLASH)) {
            Token op = previous();
            ASTNode* right = parseFactor();
            left = new BinaryOp(op, left, right);
        }
        
        return left;
    }
    
    ASTNode* parseFactor() {
        if (match(NUMBER)) {
            return new NumberLiteral(previous().value);
        }
        if (match(IDENTIFIER)) {
            return new Variable(previous().lexeme);
        }
        if (match(LPAREN)) {
            ASTNode* expr = parseExpression();
            consume(RPAREN, "Expected ')' after expression");
            return expr;
        }
        error("Expected expression");
    }
};
```

2.  **Bottom-Up Parsing (LR/LALR):**
Uses shift-reduce parsing with parse tables.

**Parser Generators:**
*   **Yacc/Bison:** LALR(1) parser generator
*   **ANTLR:** LL(*) parser generator with better error recovery

**Example (Bison):**
```yacc
%token NUMBER IDENTIFIER PLUS MINUS STAR SLASH

%%
expression:
    expression PLUS term    { $$ = new BinaryOp('+', $1, $3); }
  | expression MINUS term   { $$ = new BinaryOp('-', $1, $3); }
  | term                    { $$ = $1; }
  ;

term:
    term STAR factor        { $$ = new BinaryOp('*', $1, $3); }
  | term SLASH factor       { $$ = new BinaryOp('/', $1, $3); }
  | factor                  { $$ = $1; }
  ;

factor:
    NUMBER                  { $$ = new NumberLiteral($1); }
  | IDENTIFIER              { $$ = new Variable($1); }
  | '(' expression ')'      { $$ = $2; }
  ;
%%
```

**Abstract Syntax Tree (AST):**
```
Input: x = 2 + 3 * 4

AST:
    =
   / \
  x   +
     / \
    2   *
       / \
      3   4
```

### 🔹 Part 4: Semantic Analysis

**Purpose:**
Enforce language semantics beyond syntax.

**Tasks:**
1.  **Type Checking:** Ensure operations are type-safe.
2.  **Symbol Resolution:** Link variable uses to declarations.
3.  **Scope Management:** Enforce variable visibility rules.
4.  **Constant Folding:** Evaluate compile-time constants.

**Symbol Table:**
```cpp
class SymbolTable {
    std::map<std::string, Symbol> symbols;
    SymbolTable* parent;  // For nested scopes
    
public:
    void define(std::string name, Type type) {
        if (symbols.count(name)) {
            error("Redefinition of '" + name + "'");
        }
        symbols[name] = Symbol{type, currentScope};
    }
    
    Symbol* resolve(std::string name) {
        if (symbols.count(name)) {
            return &symbols[name];
        }
        if (parent) {
            return parent->resolve(name);
        }
        error("Undefined variable '" + name + "'");
    }
};
```

**Type Checking Example:**
```cpp
void typeCheck(ASTNode* node) {
    if (auto* binop = dynamic_cast<BinaryOp*>(node)) {
        Type leftType = typeCheck(binop->left);
        Type rightType = typeCheck(binop->right);
        
        if (leftType != rightType) {
            error("Type mismatch in binary operation");
        }
        
        if (binop->op == '+' && leftType != INT && leftType != FLOAT) {
            error("Invalid operand types for '+'");
        }
        
        return leftType;
    }
    // ... handle other node types
}
```

### 🔹 Part 5: Intermediate Representation (IR)

**Purpose:**
Platform-independent representation for optimization and code generation.

**IR Types:**

1.  **Three-Address Code (TAC):**
```
t1 = 2
t2 = 3
t3 = 4
t4 = t2 * t3
t5 = t1 + t4
x = t5
```

2.  **Static Single Assignment (SSA):**
```
x1 = 2
y1 = 3
z1 = 4
t1 = y1 * z1
t2 = x1 + t1
x2 = t2
```

**SSA Properties:**
*   Each variable assigned exactly once.
*   Enables powerful optimizations (constant propagation, dead code elimination).
*   Requires φ-functions at control flow merge points.

3.  **LLVM IR:**
```llvm
define i32 @compute(i32 %a, i32 %b) {
entry:
  %mul = mul i32 %b, 4
  %add = add i32 %a, %mul
  ret i32 %add
}
```

**IR Design Principles:**
*   **Simplicity:** Small set of operations.
*   **Uniformity:** Consistent structure.
*   **Analyzability:** Easy to reason about.
*   **Transformability:** Support optimization passes.

---

## 💻 Implementation: Simple Expression Compiler

### 🛠️ Step 1: Lexer

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cctype>

enum TokenType {
    TOK_NUMBER, TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_LPAREN, TOK_RPAREN, TOK_EOF
};

struct Token {
    TokenType type;
    int value;  // For numbers
};

class Lexer {
    std::string input;
    size_t pos = 0;
    
public:
    Lexer(const std::string& src) : input(src) {}
    
    Token nextToken() {
        while (pos < input.size() && isspace(input[pos])) pos++;
        
        if (pos >= input.size()) return {TOK_EOF, 0};
        
        char c = input[pos++];
        
        if (isdigit(c)) {
            int value = c - '0';
            while (pos < input.size() && isdigit(input[pos])) {
                value = value * 10 + (input[pos++] - '0');
            }
            return {TOK_NUMBER, value};
        }
        
        switch (c) {
            case '+': return {TOK_PLUS, 0};
            case '-': return {TOK_MINUS, 0};
            case '*': return {TOK_STAR, 0};
            case '/': return {TOK_SLASH, 0};
            case '(': return {TOK_LPAREN, 0};
            case ')': return {TOK_RPAREN, 0};
            default: throw std::runtime_error("Unknown character");
        }
    }
};
```

### 🛠️ Step 2: Parser & AST

```cpp
struct ASTNode {
    virtual ~ASTNode() = default;
    virtual int eval() = 0;
};

struct NumberNode : ASTNode {
    int value;
    NumberNode(int v) : value(v) {}
    int eval() override { return value; }
};

struct BinaryOpNode : ASTNode {
    char op;
    ASTNode *left, *right;
    
    BinaryOpNode(char o, ASTNode* l, ASTNode* r) : op(o), left(l), right(r) {}
    
    int eval() override {
        int lval = left->eval();
        int rval = right->eval();
        
        switch (op) {
            case '+': return lval + rval;
            case '-': return lval - rval;
            case '*': return lval * rval;
            case '/': return lval / rval;
            default: throw std::runtime_error("Unknown operator");
        }
    }
    
    ~BinaryOpNode() { delete left; delete right; }
};

class Parser {
    Lexer& lexer;
    Token current;
    
    void advance() { current = lexer.nextToken(); }
    
public:
    Parser(Lexer& lex) : lexer(lex) { advance(); }
    
    ASTNode* parseExpression() {
        ASTNode* left = parseTerm();
        
        while (current.type == TOK_PLUS || current.type == TOK_MINUS) {
            char op = (current.type == TOK_PLUS) ? '+' : '-';
            advance();
            ASTNode* right = parseTerm();
            left = new BinaryOpNode(op, left, right);
        }
        
        return left;
    }
    
    ASTNode* parseTerm() {
        ASTNode* left = parseFactor();
        
        while (current.type == TOK_STAR || current.type == TOK_SLASH) {
            char op = (current.type == TOK_STAR) ? '*' : '/';
            advance();
            ASTNode* right = parseFactor();
            left = new BinaryOpNode(op, left, right);
        }
        
        return left;
    }
    
    ASTNode* parseFactor() {
        if (current.type == TOK_NUMBER) {
            int value = current.value;
            advance();
            return new NumberNode(value);
        }
        
        if (current.type == TOK_LPAREN) {
            advance();
            ASTNode* expr = parseExpression();
            if (current.type != TOK_RPAREN) {
                throw std::runtime_error("Expected ')'");
            }
            advance();
            return expr;
        }
        
        throw std::runtime_error("Expected number or '('");
    }
};
```

### 🛠️ Step 3: Driver

```cpp
int main() {
    std::string input = "2 + 3 * (4 - 1)";
    
    Lexer lexer(input);
    Parser parser(lexer);
    
    ASTNode* ast = parser.parseExpression();
    int result = ast->eval();
    
    std::cout << input << " = " << result << "\n";
    
    delete ast;
    return 0;
}
```

**Output:**
```
2 + 3 * (4 - 1) = 11
```

---

## 🧪 Hands-On Labs

### Lab 85: Extend the Compiler

**Objective:** Add support for variables and assignment.

**Grammar Extension:**
```
Statement → IDENTIFIER = Expression
Expression → ...
```

**Tasks:**
1.  Add `TOK_IDENTIFIER` and `TOK_ASSIGN` tokens.
2.  Implement symbol table for variable storage.
3.  Add `AssignmentNode` and `VariableNode` AST nodes.
4.  Support multiple statements.

**Example:**
```
x = 10
y = x + 5
result = y * 2
```

---

## 📝 Summary & Key Takeaways

1.  **Compilation is Multi-Phase:** Lexing → Parsing → Semantic Analysis → IR Generation → Optimization → Code Generation.
2.  **Separation of Concerns:** Each phase has specific responsibility and well-defined interface.
3.  **Formal Methods:** Compilers rely on formal language theory (automata, grammars).
4.  **AST is Central:** Abstract Syntax Tree is the primary data structure for analysis and transformation.
5.  **IR Enables Optimization:** Platform-independent IR allows powerful, reusable optimizations.

**Compiler Complexity:**

| Phase | Input | Output | Complexity |
|---|---|---|---|
| Lexical | Characters | Tokens | $O(n)$ |
| Syntax | Tokens | AST | $O(n)$ to $O(n^3)$ |
| Semantic | AST | Annotated AST | $O(n)$ |
| IR Gen | AST | IR | $O(n)$ |
| Optimization | IR | Optimized IR | Varies |
| Code Gen | IR | Assembly | $O(n)$ |

**Real-World Compilers:**
*   **GCC:** GNU Compiler Collection (C/C++/Fortran/Ada)
*   **Clang/LLVM:** Modern modular compiler infrastructure
*   **MSVC:** Microsoft Visual C++ Compiler
*   **Rust Compiler (rustc):** Uses LLVM backend

---

## 📚 Additional Resources

*   [Dragon Book: Compilers: Principles, Techniques, and Tools](https://www.pearson.com/store/p/compilers-principles-techniques-and-tools/P100000843907)
*   [Engineering a Compiler (Cooper & Torczon)](https://www.elsevier.com/books/engineering-a-compiler/cooper/978-0-12-088478-0)
*   [LLVM Tutorial](https://llvm.org/docs/tutorial/)

**Tomorrow:** Day 86 - Lexical Analysis Deep Dive... regular expressions, DFA construction, and lexer optimization.

*End of Day 085 - Total Lines: 1000+*
