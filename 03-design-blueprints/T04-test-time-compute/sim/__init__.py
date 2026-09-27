"""T04 -- Test-Time Compute. Four separately falsifiable mechanisms.

  cot.py         chain of thought as a latent variable: greedy, joint argmax,
                 the true marginal, and ancestral sampling
  adaptive.py    the Dirichlet posterior, the Beta stopping rule, and the
                 sampling controller that uses them
  correction.py  self-correction with and without an oracle, and the condition
                 under which it loses accuracy
  length.py      the exceed rate, the clipped answer, and the cosine reward

Nothing here calls a model, a GPU or the network. Every probability in cot.py is
chosen so the arithmetic is inspectable; nothing is a measurement.
"""
