.. _intro:

************
Introduction
************

HMIP is a free software package for convex optimization based on the
Python programming language.  It can be used with the interactive Python
interpreter, on the command line by executing Python scripts, or integrated
in other software via Python extension modules.  Its main purpose is to
make the development of software for convex optimization applications
straightforward by building on Python's extensive standard library and on
the strengths of Python as a high-level programming language.

HMIP is organized in different modules.

:mod:`cvxopt.solvers <cvxopt.solvers>`
  Convex optimization routines and optional interfaces to solvers from
  GLPK, MOSEK, and DSDP5 (:ref:`c-coneprog` and :ref:`c-solvers`).

:mod:`cvxopt.modeling <cvxopt.modeling>`
  Routines for specifying and solving linear programs and convex
  optimization problems with piecewise-linear cost and constraint functions
  (:ref:`c-modeling`).

:mod:`cvxopt.info <cvxopt.info>`
  Defines a string :const:`version` with the version number of the CVXOPT
  installation and a function :func:`license` that prints the CVXOPT
  license.

:mod:`cvxopt.printing <cvxopt.printing>`
  Contains functions and parameters that control how matrices are formatted.

The modules are described in detail in this manual and in the on-line Python
help facility :program:`pydoc`.  Several example scripts are included in
the distribution.
