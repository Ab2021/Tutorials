"""T01 sampling-decoding simulation package.

A model of the *mechanism* of next-token sampling: how temperature reshapes a
distribution, what each truncation rule keeps, why processor order matters, and
how tiny floating-point perturbations move an argmax.

Nothing in this package is a measurement of a real model or a real GPU.
See `sim/distribution.py` for the shape assumptions.
"""
