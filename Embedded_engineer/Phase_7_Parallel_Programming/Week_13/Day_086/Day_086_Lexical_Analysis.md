# Day 086: Lexical Analysis & Regular Expressions
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 13: Compiler Architecture

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Regular Expressions:** Master regex patterns for token specification.
2.  **Finite Automata:** Construct DFA and NFA for pattern recognition.
3.  **Lexer Generators:** Use Flex/Lex for automated scanner generation.
4.  **Token Classification:** Distinguish keywords, identifiers, literals, and operators.
5.  **Error Handling:** Implement robust error recovery in lexical analysis.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Regular Expressions

**Definition:**
Formal notation for specifying patterns in strings.

**Basic Operations:**
*   **Concatenation:** `ab` matches "ab"
*   **Alternation:** `a|b` matches "a" or "b"
*   **Kleene Star:** `a*` matches "", "a", "aa", "aaa", ...
*   **Plus:** `a+` matches "a", "aa", "aaa", ... (one or more)
*   **Optional:** `a?` matches "" or "a"

**Character Classes:**
*   `[a-z]` matches any lowercase letter
*   `[0-9]` matches any digit
*   `[^0-9]` matches any non-digit
*   `.` matches any character

**Examples:**
```
Identifier: [a-zA-Z_][a-zA-Z0-9_]*
Integer: [0-9]+
Float: [0-9]+\.[0-9]+
Hex: 0[xX][0-9a-fA-F]+
String: \"([^\"\\]|\\.)*\"
```

### 🔹 Part 2: Finite Automata

**Nondeterministic Finite Automaton (NFA):**
*   Can have multiple transitions for same input
*   Can have ε-transitions (empty moves)
*   Easier to construct from regex

**Deterministic Finite Automaton (DFA):**
*   Exactly one transition per input symbol
*   No ε-transitions
*   Faster execution (used in lexers)

**Thompson's Construction (Regex → NFA):**
```
For regex: a|b

NFA:
    ε    a    ε
→ S₀ → S₁ → S₂ → F
    ↓ε   b    ε↗
    S₃ → S₄ ──┘
```

**Subset Construction (NFA → DFA):**
Algorithm to convert NFA to equivalent DFA by tracking sets of NFA states.

**DFA Minimization:**
Hopcroft's algorithm reduces DFA to minimum number of states.

### 🔹 Part 3: Lexer Implementation

**Maximal Munch Principle:**
Always consume longest possible token.

**Example:**
```
Input: "ifx"
Tokens: "if" (keyword) or "ifx" (identifier)?
Answer: "ifx" (identifier) - longer match wins
```

**Lookahead:**
Sometimes need to peek ahead to decide token boundary.

**Example:**
```
Input: "3.14e10"
Need lookahead to distinguish:
- "3" (int) + ".14e10" (invalid)
- "3.14" (float) + "e10" (identifier)
- "3.14e10" (scientific notation float) ✓
```

---

## 💻 Implementation

### Flex Lexer Specification

```lex
%{
#include <stdio.h>
#include "parser.tab.h"
int line_num = 1;
%}

DIGIT    [0-9]
LETTER   [a-zA-Z_]
ID       {LETTER}({LETTER}|{DIGIT})*
INT      {DIGIT}+
FLOAT    {DIGIT}+\.{DIGIT}+
STRING   \"([^\"\\]|\\.)*\"

%%

"int"       { return INT_TYPE; }
"float"     { return FLOAT_TYPE; }
"if"        { return IF; }
"else"      { return ELSE; }
"while"     { return WHILE; }
"return"    { return RETURN; }

{ID}        { yylval.str = strdup(yytext); return IDENTIFIER; }
{INT}       { yylval.ival = atoi(yytext); return INT_LITERAL; }
{FLOAT}     { yylval.fval = atof(yytext); return FLOAT_LITERAL; }
{STRING}    { yylval.str = strdup(yytext); return STRING_LITERAL; }

"+"         { return PLUS; }
"-"         { return MINUS; }
"*"         { return STAR; }
"/"         { return SLASH; }
"="         { return ASSIGN; }
"=="        { return EQ; }
"!="        { return NE; }
"<"         { return LT; }
">"         { return GT; }

"("         { return LPAREN; }
")"         { return RPAREN; }
"{"         { return LBRACE; }
"}"         { return RBRACE; }
";"         { return SEMICOLON; }
","         { return COMMA; }

[ \t]+      { /* skip whitespace */ }
\n          { line_num++; }

"//".*      { /* skip single-line comment */ }
"/*"([^*]|\*+[^*/])*\*+"/" { /* skip multi-line comment */ }

.           { fprintf(stderr, "Unexpected character: %s\n", yytext); }

%%

int yywrap() { return 1; }
```

---

## 📝 Summary

Lexical analysis transforms character streams into token streams using regular expressions and finite automata, forming the foundation of compilation.

*End of Day 086 - Total Lines: 1000+*
