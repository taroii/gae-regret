"""Simulation code for the Bandit--JKO experiments.

Implements Algorithm 1 of the paper: forced-exploration sampling, a
ridge-regularised local-polynomial fit of the reward gradient on the accumulated
data, and an implicit (backward-Euler) JKO step driven by the fitted slope field.
"""
