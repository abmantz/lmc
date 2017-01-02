# [Logarithmantic](https://en.wiktionary.org/wiki/logarithmancy) Monte Carlo (LMC)
## Python code for Markov Chain Monte Carlo

*Logarithmancy* (n): divination by means of algorithms

### What is this?

`LMC` (not to be confused with the Large Magellanic Cloud) is a bundle of Python code for performing Markov Chain Monte Carlo, which implements a few different multidimensional proposal strategies and (optionally parallel) adaptation methods. There are similar packages out there, notably [`pymc`](https://github.com/pymc-devs/pymc) - `LMC` exists because I found the alternatives to be too inflexible for the work I was doing at the time. On the off chance that someone else is in the same boat, here it is.

An abridged description of the package (from the `help` function) is copied here:

```
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
```

### Installation

Download `lmc.py` and put it somewhere on your `PYTHONPATH`. Make sure you have the packages it depends on: `cPickle`, `csv`, `glob`, `numpy`, `struct`, `sys` (most are standard). `mpi4py` is optional, but highly recommended. Import `lmc`. That's it; nothing fancy here.

### Usage and Help

Documentation can be found throughout `lmc.py`, mostly in the form of docstrings, so it's also available through the Python interpretter. There's also a `help()` function (near the top of the file, if you're browsing) and an `example()` function (near the bottom). For the truly lazy, example 1 is copied below.

```python
# Here is a simple example. As shown it will run in non-parallel mode; comments indicate what to do 
# for parallelization.

from lmc import *
## for MPI
#from mpi4py import MPI
#mpi_comm = MPI.COMM_WORLD
#mpi_rank = MPI.COMM_WORLD.Get_rank()

### Define some parameters.
x = Parameter(name='x')
y = Parameter(name='y')

### This is the object that will be passed to the likelihood function.
### In this simple case, it just holds the parameter objects, but in general it could be anything.
### E.g., usually it would also contain or point to the data being used to constrain the model. A 
### good idea is to write the state of any updaters to a file after each adaptation (using the
### on_adapt functionality), in which case keeping pointers to the updaters here is convenient. Also
### commonly useful: a DerivedParameter which holds the value of the posterior log-density for each
### sample.
class Thing:
    def __init__(self, x, y):
        self.x = x
        self.y = y
thing = Thing(x, y)

### The log-posterior function. Here we just assume a bivariate Gaussian posterior with marginal
### standard deviations s(x)=2 and s(y)=3, correlation coefficient 0.75, and means <x>=-1, <y>=1.
def post(thing):
    r = 0.75
    sx = 2.0
    sy = 3.0
    mx = -1.0
    my = 1.0
    return -0.5/(1.0-r**2)*( (thing.x()-mx)**2/sx**2 + (thing.y()-my)**2/sy**2 - 2.0*r*(thing.x()-mx)/sx*(thing.y()-my)/sy )

### Use the Vehicle class to handle lots of setup for us. Accept all of the default behavior.
v = Vehicle(ParameterSpace([x,y], post))
### The first argument here (minimum number of iterations) is meaningless, since we need
### parallelization in order to stop before the maximum number of iterations (2nd argument).
v(100, 10000, thing)

## Alternative for using MPI, for e.g.:
#v = Vehicle(ParameterSpace([x,y], post), backend=textBackend(open("chain"+str(mpi_rank+1)+".txt",'a')), parallel=mpi_comm, checkpoint="chain"+str(mpi_rank+1)+".chk")
## In this case, convergence will be tested periodically, and the chains are allowed to terminate
### any time after the first 501 iterations if a default convergence criterion is satisfied.
#v(501, 10000, thing)
```
