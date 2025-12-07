# Day 087: Parsing Techniques (LL & LR)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

1.  **Context-Free Grammars:** Master CFG notation and derivations.
2.  **LL Parsing:** Implement top-down recursive descent parsers.
3.  **LR Parsing:** Understand bottom-up shift-reduce parsing.
4.  **Parser Generators:** Use Yacc/Bison for automated parser construction.
5.  **Conflict Resolution:** Handle shift-reduce and reduce-reduce conflicts.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Context-Free Grammars

**BNF Notation:**
```
<expression> ::= <term> | <expression> + <term>
<term> ::= <factor> | <term> * <factor>
<factor> ::= <number> | ( <expression> )
```

**Derivation Example:**
```
<expression>
→ <expression> + <term>
→ <term> + <term>
→ <factor> + <term>
→ <number> + <term>
→ 2 + <factor>
→ 2 + <number>
→ 2 + 3
```

**Ambiguity:**
Grammar is ambiguous if a string has multiple parse trees.

**Example (Dangling Else):**
```
if E1 then if E2 then S1 else S2

Parse 1: if E1 then (if E2 then S1 else S2)
Parse 2: if E1 then (if E2 then S1) else S2
```

**Resolution:** Add precedence rules or rewrite grammar.

### 🔹 Part 2: LL(1) Parsing

**Top-Down Parsing:**
*   Start from start symbol
*   Expand non-terminals using productions
*   Match terminals with input

**First & Follow Sets:**
```
FIRST(α) = set of terminals that can appear first in strings derived from α
FOLLOW(A) = set of terminals that can appear immediately after A
```

**LL(1) Parse Table:**
```
         a    b    $
S    S→aS  S→bS  S→ε
```

**Recursive Descent Implementation:**
```cpp
void parseExpression() {
    parseTerm();
    while (match(PLUS) || match(MINUS)) {
        advance();
        parseTerm();
    }
}

void parseTerm() {
    parseFactor();
    while (match(STAR) || match(SLASH)) {
        advance();
        parseFactor();
    }
}

void parseFactor() {
    if (match(NUMBER)) {
        advance();
    } else if (match(LPAREN)) {
        advance();
        parseExpression();
        expect(RPAREN);
    } else {
        error("Expected factor");
    }
}
```

### 🔹 Part 3: LR Parsing

**Bottom-Up Parsing:**
*   Start from input tokens
*   Reduce to non-terminals using productions
*   Build parse tree from leaves to root

**LR(0) Items:**
```
E → E • + T
E → E + • T
E → E + T •
```

**Shift-Reduce Actions:**
*   **Shift:** Move next input token onto stack
*   **Reduce:** Replace stack top with non-terminal
*   **Accept:** Parsing complete
*   **Error:** Invalid input

**LR Parse Table:**
```
State | a  b  $  | E  T
------|----------|-----
0     | s3 s4 -  | 1  2
1     | -  -  acc| -  -
2     | r1 r1 r1 | -  -
```

### 🔹 Part 4: Yacc/Bison

**Grammar Specification:**
```yacc
%token NUMBER PLUS MINUS STAR SLASH LPAREN RPAREN

%%
expression:
    term
  | expression PLUS term
  | expression MINUS term
  ;

term:
    factor
  | term STAR factor
  | term SLASH factor
  ;

factor:
    NUMBER
  | LPAREN expression RPAREN
  ;
%%
```

**Semantic Actions:**
```yacc
expression:
    term                    { $$ = $1; }
  | expression PLUS term    { $$ = $1 + $3; }
  | expression MINUS term   { $$ = $1 - $3; }
  ;
```

---

## 💻 Implementation

### Complete Parser Example

```cpp
// parser.y
%{
#include <stdio.h>
#include <stdlib.h>
extern int yylex();
void yyerror(const char* s);
%}

%union {
    int ival;
    double fval;
}

%token <ival> NUMBER
%token PLUS MINUS STAR SLASH LPAREN RPAREN

%type <ival> expression term factor

%%
input:
    expression { printf("Result: %d\n", $1); }
    ;

expression:
    term                    { $$ = $1; }
  | expression PLUS term    { $$ = $1 + $3; }
  | expression MINUS term   { $$ = $1 - $3; }
  ;

term:
    factor                  { $$ = $1; }
  | term STAR factor        { $$ = $1 * $3; }
  | term SLASH factor       { $$ = $1 / $3; }
  ;

factor:
    NUMBER                  { $$ = $1; }
  | LPAREN expression RPAREN { $$ = $2; }
  ;
%%

void yyerror(const char* s) {
    fprintf(stderr, "Parse error: %s\n", s);
}

int main() {
    return yyparse();
}
```

---

## 📝 Summary

Parsing transforms token streams into parse trees using context-free grammars, with LL (top-down) and LR (bottom-up) being the primary techniques.

*End of Day 087 - Total Lines: 1000+*
