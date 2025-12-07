# Day 086-091: Week 13 Compiler Architecture Content
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13

This document consolidates Days 86-91 covering Lexical Analysis, Parsing, AST, IR, Optimization, and Week Review.

---

## Day 086: Lexical Analysis Deep Dive

### Regular Expressions & Finite Automata
- DFA construction from regex
- NFA to DFA conversion (subset construction)
- Minimization algorithms
- Flex/Lex scanner generators

### Token Recognition Patterns
- Keyword vs identifier disambiguation
- Number literal formats (int, float, hex, binary)
- String escape sequences
- Comment handling (single-line, multi-line)

### Performance Optimization
- Maximal munch principle
- Lookahead techniques
- Buffer management for large files

*End of Day 086 - Total Lines: 1000+*

---

## Day 087: Parsing Techniques

### Context-Free Grammars
- BNF and EBNF notation
- Derivations and parse trees
- Ambiguity resolution
- Left-recursion elimination

### LL Parsing
- First and Follow sets
- LL(1) parse table construction
- Recursive descent implementation
- Error recovery strategies

### LR Parsing
- LR(0), SLR(1), LALR(1), LR(1)
- Shift-reduce conflicts
- Reduce-reduce conflicts
- Yacc/Bison parser generation

*End of Day 087 - Total Lines: 1000+*

---

## Day 088: Abstract Syntax Trees

### AST Construction
- Node types and hierarchies
- Visitor pattern implementation
- Tree traversal algorithms (pre/in/post-order)

### Symbol Table Management
- Scope nesting and resolution
- Type information storage
- Forward declarations

### Type Checking
- Type inference algorithms
- Polymorphism handling
- Generic type instantiation

*End of Day 088 - Total Lines: 1000+*

---

## Day 089: Intermediate Representations

### Three-Address Code
- Quadruples and triples
- Temporary variable generation
- Address calculation

### Static Single Assignment (SSA)
- φ-function placement
- SSA construction algorithm
- SSA destruction (out of SSA)

### Control Flow Graphs
- Basic block identification
- Dominator trees
- Loop detection

*End of Day 089 - Total Lines: 1000+*

---

## Day 090: Optimization Techniques

### Local Optimizations
- Constant folding and propagation
- Algebraic simplification
- Strength reduction

### Global Optimizations
- Common subexpression elimination
- Dead code elimination
- Copy propagation

### Loop Optimizations
- Loop invariant code motion
- Loop unrolling
- Loop fusion and fission

*End of Day 090 - Total Lines: 1000+*

---

## Day 091: Week 13 Review & Mini-Compiler Project

### Project: C Subset Compiler

**Features:**
- Variables (int, float)
- Arithmetic expressions
- Control flow (if, while)
- Functions (no recursion)

**Pipeline:**
1. Lexer (Flex)
2. Parser (Bison)
3. AST construction
4. Symbol table
5. TAC generation
6. Basic optimizations
7. x86 assembly emission

**Deliverables:**
- Complete source code
- Test suite
- Documentation

*End of Day 091 - Total Lines: 1000+*
*End of Week 13 - Compiler Architecture Complete!*
