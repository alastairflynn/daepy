.. DAEpy documentation master file, created by
   sphinx-quickstart on Wed Oct  2 14:30:07 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
=================================

DAEpy is a Python library for solving boundary value problems of differential algebraic equations with advanced and retarded (forward and backward) delays. It also contains routines for parameter continuation. The :download:`numerical method <summary.pdf>` is based on collocation. The `source code <https://github.com/alastairflynn/daepy>`_ is available on Github.

This library was developed at `LCVMM <https://lcvmwww.epfl.ch/>`_. If you find it useful, please cite it [:download:`BibTeX <DAEpy.bib>`]:

  Alastair Flynn. DAEpy: a Python library for solving boundary value problems of differential algebraic equations with advanced and retarded (forward and backward) delays. version 1.0.1. LCVMM, EPFL. 2019. url: https://lcvmwww.epfl.ch/software/daepy/.

Installation
============

DAEpy can be installed using pip. ::

  pip install daepy

This will install DAEpy and all its dependencies. It is recommended, but not necessary, to also install scikit-umfpack which contains a routine for solving sparse linear systems. ::

  pip install scikit-umfpack

Usage
================

The user must define their system by a Python class. A template class can be imported by ::

  from daepy import DAETemplate

A problem definition class should look as follows: ::

  class DAE():
      def __init__(self, parameter):
          self.N = 2
          self.dindex = [0]
          self.aindex = [1]
          self.parameter = parameter

      def fun(self, x, y, transform):
          ...
          return f

      def bv(self, y, transform):
          ...
          return b

      def jacobian(self, x, y, transform):
          ...
          return dfdy, dfdt

      def bv_jacobian(self, y, transform):
          ...
          return dbdy, dbdt

      def update_parameter(self, p):
          self.parameter = p

      def parameter_jacobian(self, x, y):
          ...
          return dfdp, dbdp

Every problem definition must define the attributes

* :attr:`N` the dimension of the system
* :attr:`dindex` the indices of the differential variables
* :attr:`aindex` the indices of the algebraic variables

and the methods

* :meth:`.fun` which calculates the residual of the system
* :meth:`.bv` which calculates the boundary conditions

The class may also define the methods

* :meth:`.jacobian` which calculates the jacobian of the system
* :meth:`.bv_jacobian` which calculates the jacobian of the boundary conditions
* :meth:`.update_parameter` which updates a system parameter
* :meth:`.parameter_jacobian` which calculates the jacobian of the system and boundary conditions with respect to a system parameter

All six methods must be defined for parameter continuation. You can of course define your own attributes and methods as well. The :class:`.BVPSolution` class has several methods to aid construction of jacobians.

.. note::
  The system must reduced to first order. *Differential variables* are variables whose derivative appears in the system and *algebraic variables* are variables whose derivative does not appear. See :download:`numerical method <summary.pdf>` for more details.

Once a problem definition has been written, the :class:`.BVP` class is used to construct and solve the nonlinear system ::

  from daepy import BVP
  from mydae import DAE # problem definition saved as mydae.py

  parameter = 2.0
  dae = DAE(parameter)

  bvp = BVP(dae, degree=3, intervals=10)
  bvp.initial_guess([lambda x: 0, lambda x: 0], initial_interval=[0,1])

  sol = bvp.solve()

The solution is a :class:`.BVPSolution` object. A continuation run can be performed using the :meth:`.continuation` method (it is not necessary to call :meth:`.solve` before :meth:`.continuation`).

Examples
========

There is a :ref:`basic usage example <basic_example>` and a :ref:`parameter continuation <continuation_example>` example.

Reference
=========

.. toctree::
   :maxdepth: 0

   bvp
   dae
   collocation
   continuation
   nonlinear
   derivatives

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
