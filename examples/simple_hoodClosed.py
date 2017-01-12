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
