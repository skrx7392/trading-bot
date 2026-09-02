"""Scheduled entrypoints.

Nothing in this package computes anything. Each module here is a *sequence* of
calls into the library, plus the summary an operator reads the next morning —
so that an unattended run has exactly one place where its order, its dates and
its failure modes are decided.
"""
