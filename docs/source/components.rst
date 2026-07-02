.. _core_components:

===============
Core Components
===============

These pages document the building blocks that power quantifiers and
optimization workflows, grouped by role.

Building blocks of a quantifier
===============================

The pieces a matching quantifier composes — a *representation*, a *loss*, and a
*solver* — plus the compose framework that wires them together.

.. toctree::
   :maxdepth: 1

   modules/representations.rst
   modules/losses.rst
   modules/solvers.rst
   modules/compose.rst

Multiclass and output handling
==============================

How binary methods scale to multiclass problems, and how the resulting prevalence
estimates are normalised and formatted.

.. toctree::
   :maxdepth: 1

   modules/multiclass.rst
   modules/prevalence_normalization.rst

Performance
===========

The optional compiled acceleration behind distribution matching.

.. toctree::
   :maxdepth: 1

   modules/cython_acceleration.rst
