"""T03 constrained-generation simulation package.

Three mechanisms, each isolated:

  * automata.py -- schema to DFA, and the nesting boundary where a DFA stops
    working and a pushdown automaton is required.
  * masking.py  -- the mask itself: -inf before softmax, renormalise, and the
    sparsity that makes the whole approach worth doing.
  * healing.py  -- token healing: roll back, require the next token to start
    with the predicted prefix, preserve the surface form.
  * fudge.py    -- semantic constraints over a truncated candidate set.

This is a MODEL OF A MECHANISM over a toy vocabulary. It is not a measurement
of any real engine, and the counts it prints are its own. The corpus's figures
(~10 legal tokens of ~100,000; the top-200 FUDGE truncation) are quoted and
attributed in HLD.md section 9 and are NOT reproduced here.
"""
