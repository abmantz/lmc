.. image:: https://img.shields.io/badge/ascl-1706.005-blue.svg?colorB=262255
   :alt: ascl:1706.005
   :target: http://ascl.net/1706.005
.. image:: https://img.shields.io/pypi/v/lmc.svg
   :alt: PyPi
   :target: https://pypi.python.org/pypi/lmc
.. image:: https://img.shields.io/pypi/l/lmc.svg
   :alt: LGPL-3.0
   :target: https://www.gnu.org/licenses/lgpl-3.0.txt

=====================================================================================
Logarithmantic Monte Carlo (LMC)
=====================================================================================

----------------------------------------
Python code for Markov Chain Monte Carlo
----------------------------------------

`Logarithmancy <https://en.wiktionary.org/wiki/logarithmancy>`_ (n): divination by means of algorithms

What is this?
=============

``LMC`` (not to be confused with the Large Magellanic Cloud) is a bundle of Python code for performing Markov Chain Monte Carlo, which implements a few different multidimensional proposal strategies and (optionally parallel) adaptation methods. There are similar packages out there, notably `pymc <https://github.com/pymc-devs/pymc>`_ - ``LMC`` exists because I found the alternatives to be too inflexible for the work I was doing at the time. On the off chance that someone else is in the same boat, here it is.

The samplers currently included are Metropolis, slice, and the affine-invariant sampler popularized by `emcee <http://dan.iel.fm/emcee>`_ (`Goodman & Weare 2010 <http://dx.doi.org/10.2140/camcos.2010.5.65>`_).

An abridged description of the package (from the `help` function) is copied here::

 The module should be very flexible, but is designed with these things foremost in mind:
  1. use with expensive likelihood calculations which probably have a host of hard-to-modify
     code associated with them.
  2. making it straightforward to break the parameter space into subspaces which can be sampled
     using different proposal methods and at different rates. For example, if changing some
     parameters requires very expensive calulations in the likelihood, the other, faster
     parameters can be sampled at a higher rate. Or, some parameters may lend themselves to
     Gibbs sampling, while others may not, and these can be block updated independently.
  3. keeping the overhead low to facilitate large numbers of parameters. Some of this has been
     lost in the port from C++, but, for example, the package provides automatic tuning of the
     proposal covariance for block updating without needing to store traces of the parameters in
     memory.

 Real-valued parameters are usually assumed, but the framework can be used with other types of 
 parameters, with suitable overloading of classes.

 A byproduct of item (1) is that the user is expected to handle all aspects of the calculation of 
 the posterior. The module doesn't implement assignment of canned, standard priors, or automatic 
 discovery of shortcuts like conjugate Gibbs sampling. The idea is that the user is in the best 
 position to know how the details of the likelihood and priors should be implemented.

 Communication between parallel chains can significantly speed up convergence. In parallel mode, 
 adaptive Updaters use information from all running chains to tune their proposals, rather than 
 only from their own chain. The Gelman-Rubin convergence criterion (ratio of inter- to intra-chain 
 variances) for each free parameter is also calculated. Parallelization is implemented in two ways; 
 see ?Updater for instructions on using each.
  1. Via MPI (using mpi4py). MPI adaptations are synchronous: when a chain reaches a communication
     point, it stops until all chains have caught up.
  2. Via the filesystem. When a chain adapts, it will write its covariance information to a file. It
     will then read in any information from other chains that is present in similar files, and
     incorporate it when tuning. This process is asynchronous; chains will not wait for one another; 
     they will simply adapt using whatever information has been shared at the time. 


Installation
============

Automatic
---------

Install from PyPI by running ``pip install lmc``.

Manual
------

Download ``lmc/lmc.py`` and put it somewhere on your ``PYTHONPATH``. You will need to have the ``numpy`` package installed. The ``mpi4py`` package is optional, but highly recommended.

Usage and Help
==============

Documentation can be found throughout ``lmc.py``, mostly in the form of docstrings, so it's also available through the Python interpreter. There's also a ``help()`` function (near the top of the file, if you're browsing) and an ``example()`` function (near the bottom).

The examples can also be browsed `here <https://github.com/abmantz/lmc/tree/master/examples>`_.
