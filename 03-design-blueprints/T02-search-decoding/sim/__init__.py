"""T02 search-decoding simulation package.

One priority-queue decoder, parameterised by (comparator, beam constraint K,
heuristic h). Greedy, beam search, uniform-cost and A* are four settings of
that one machine -- which is the unifying result the topic rests on.

This is a MODEL OF A MECHANISM over a toy bigram language model. It is not a
measurement of any real decoder, and the step counts it prints are its own.
"""
