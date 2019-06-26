"""
A module for Markov Chain Monte Carlo. See help() and example().
Copyright (C) 2011 Adam Mantz
Licensed under the  GNU LESSER GENERAL PUBLIC LICENSE
"""

import csv
import pickle
import glob
import numpy as np
import struct
import sys
try:
    from mpi4py import MPI
except ImportError:
    pass
# try:
#     import numdifftools
#     have_numdifftools = True
# except ImportError:
#     have_numdifftools = False
try:
    import scipy.optimize
    have_scipy = True
except ImportError:
    have_scipy = False

parallel_filename_base = 'lmc'
parallel_filename_ext = '.chk'


def help():
    print("""
The module should be very flexible, but is designed with these things foremost in mind:
  1. use with expensive likelihood calculations which probably have a host of hard-to-modify code associated with them.
  2. making it straightforward to break the parameter space into subspaces which can be sampled using different proposal methods and at different rates. For example, if changing some parameters requires very expensive calulations in the likelihood, the other, faster parameters can be sampled at a higher rate. Or, some parameters may lend themselves to Gibbs sampling, while others may not, and these can be block updated independently.
  3. keeping the overhead low to facilitate large numbers of parameters. Some of this has been lost in the port from C++, but, for example, the package provides automatic tuning of the proposal covariance for block updating without needing to store traces of the parameters in memory.

Real-valued parameters are usually assumed, but the framework can be used with other types of parameters, with suitable overloading of classes.

A byproduct of item (1) is that the user is expected to handle all aspects of the calculation of the posterior. The module doesn't implement assignment of canned, standard priors, or automatic discovery of shortcuts like conjugate Gibbs sampling. The idea is that the user is in the best position to know how the details of the likelihood and priors should be implemented. (Note that, as is, the code very naturally handles log-posterior values of -infinity, corresponding to zero probability, e.g. when a proposed parameter value is outside the allowed range. The only point to be careful of is that most samplers need to start at a point with non-zero probability.)

It's probably worth reading the docstrings for:
 Parameter
 ParameterSpace
 Updater, CartesianSequentialUpdater, CartesianPermutationUpdater, MultiDimSequentialUpdater, MultiDimPermutationUpdater, MultiDimRotationUpdater, GoodmanWeareUpdater, TableUpdater
 Slice, Metropolis 
 randNormalExp, randChiExp
 textBackend, stdoutBackend, dictBackend, binaryBackend
 Engine
 Vehicle
 

Here is a quick overview of the class structure:
 Parameter objects should be self explanatory.
 ParameterSpace objects represent sets of Parameters for organizational purposes.
 Updater objects provide the mechanism to sample a ParameterSpace. They come in several varieties:
   * Cartesian updaters perform updates to each Parameter in their ParameterSpace individually.
   * MultiDim updaters perform block updates to all parameters in their ParameterSpace at the same time.
  Each of these comes in Sequential and Permutation flavors, corresponding to sampling each direction in the ParameterSpace in fixed or random order. There is also a Rotation version of the MultiDim updater, which proposes along completely random directions in the multi-dimension parameter space, with the step length scaled appropriately for the direction chosen, given the covariance matrix.
  Updaters can adapt their proposal distributions. Cartesian updaters tune the typical scale of proposed steps for each parameter, while MultiDim updaters tune the set of basis directions used to explore the ParameterSpace, as well as the scale along each direction.
  There are some more specialized Updaters, with somewhat restricted functionality compared to the very general ones above.
   * GoodmanWeareUpdater is an ensemble updater (i.e. does not generate a Markov chain); see its docstring.
   * TableUpdater allows you to restrict some parameters to a particular set of tabulated values, and uses the Metropolis stepper to try to move among them.
  Updater states (e.g. the proposal basis after adapting) can be saved to and restored from files using the save() and restore() methods. The relevant data are stored as a dictionary using cPickle; this should be safer and preferable to pickling the Updater object directly. Cartesian and MultiDim updaters are mutually compatible as far as this functionality is concerned, inasmuch as that's possible.
 Step objects implement the specific algorithm used to propose a step along a given direction. Slice and Metropolis algorithms are implemented. The distribution of proposal lengths by Metropolis Step objects is customizable.
 Backend objects handle the storage of Parameter values as the chain progresses.
 Engine objects hold a list of Updater objects, each of which is called in a single iteration of the chain.
 Vehicle objects encapsulate most of the setup necessary for the most common analysis situation (a single parameter space and updater).

Finally, the ChiSquareLikelihood class may simplify the setup of least-squares style problems. Similar helper classes could clearly be added.


Communication between parallel chains can significantly speed up convergence. In parallel mode, adaptive Updaters use information from all running chains to tune their proposals, rather than only from their own chain. The Gelman-Rubin convergence criterion (ratio of inter- to intra-chain variances) for each free parameter is also calculated. Parallelization is implemented in two ways; see ?Updater for instructions on using each.
  1. Via MPI (using mpi4py). MPI adaptations are synchronous: when a chain reaches a communication point, it stops until all chains have caught up. All Updaters in a given chain should use the same communicator (at least, there's no advantage in doing otherwise). (This could be made safely asynchronous using shared memory, although I'm not sure that would be an improvement.)
  2. Via the filesystem. When a chain adapts, it will write its covariance information to a file. It will then read in any information from other chains that is present in similar files, and incorporate it when tuning. This process is asynchronous; chains will not wait for one another, they will simply adapt using whatever information has been shared at the time. The global variables parallel_filename_base and parallel_filename_ext can be used to customize the prefix and suffix of the files written.


The definition of an MCMC iteration in this implementation can be a little confusing. As far as an MultiDim updater object's 'count' attribute (controlling the timing of adaptations, for example) is concerned, an iteration corresponds to one proposal along some direction. For Cartesian updaters, each iteration corresponds to a separate proposal for all of the Parameters in the Updater's ParameterSpace. (This disparity is unfortunate, but ensures that the count attribute is the correct value to use when, e.g., calculating chain variances on the fly.) In either case, the Updater's rate attribute determines how many of these iterations occur each time the updater is called by the Engine (default=1). However, as far as the Engine is concerned, an iteration corresponds to a loop over all loaded Updaters. So if there are two MultiDim updaters, u1 and u2, with rates set to 1 and 2, and the Engine's updater list is [u2, u1, u2] (repetition is perfectly allowed, and sometimes desirable), then each iteration of the Engine actually corresponds to 1*u1.rate + 2*u2.rate = 5 proposals. However, if u1 is instead a Cartesian updater controlling 3 Parameters, then each engine iteration implies 7 proposals. The chain is stored to the Backend(s) after each Engine iteration. The upshot of all this is that manipulating the Engine's list of Updaters provides lots of flexibility to sample parameters at different rates and/or thin the chain as it's being run, at the expense of a little complexity.

See also lmc.example()


Crude non-MCMC functionality:
1. ParameterSpace.optimize provides deterministic minimization (using scipy.optimize.fmin_powell) to get an decent initial fit. This is not very efficient.
2. Updater.set_covariance_from_hessian uses finite differencing to estimate an appropriate cartesian width in each direction. This will fail if the state is not in a local minimum, or just because. (NB commented due to compatibility issues.)
""")




class Parameter:
    """
    Class to handle a single free parameter. Can/often should be overriden. The only critical attributes are:
     1. width: initial guess for step lengths. Adaptive CartesianUpdaters change this value.
     2. (): return the current parameter value.
     3. set( ): set the parameter to a new value.
    """
    def __init__(self, value=0.0, width=1.0, name=''):
        self.value = value
        self.width = width
        self.name = name
    def __call__(self):
        return self.value
    def __str__(self):
        return self.name + ': ' + str(self.value) + ' ' + str(self.width)
    def set(self, value):
        self.value = value


class DerivedParameter(Parameter):
    """
    Prototype of a Parameter whose value may not be sampled because it is a deterministic function of other parameters. The user is responsible for setting its value. This should be done directly rather than by using set(), for safety reasons.
    """
    def __init__(self, value=0.0, name=''):
        Parameter.__init__(self, value, None, name)
    def set(self, value):
        raise Exception('Attempt to set() value of a DerivedParameter.')


class postgetter:
    # needs space and struct to be defined
    # actually returns -2*loglike
    def __init__(self):
        self.verbose = False
        self.last = 1e300
    def __call__(self, x):
        for i, p in enumerate(self.space):
            self.space[i].set(x[i])
        chisq = -2.0 * self.space.log_posterior(self.struct)
        if np.isinf(chisq):
            chisq = 1e300
        if self.verbose and chisq < self.last:
            self.last = chisq
            print(chisq, x)
        return chisq

class ParameterSpace(list):
    """
    Class to define sets of parameters (parameter spaces); inherits list. To sample the parameter space, attribute log_posterior must be set to a function of one argument that evaluates the *complete* posterior likelihood, including priors and parameters not in this ParameterSpace.
    """
    def __init__(self, parameterList=[], log_posterior=None):
        list.__init__(self, parameterList)
        self.log_posterior = log_posterior
    def __str__(self):
        st = ''
        for p in self:
            st = st + '\n' + str(p)
        return st
    def optimize(self, struct, xtol=0.01, ftol=0.01, maxiter=10):
        if not have_scipy:
            print("ParameterSpace.optimize requires the scipy package -- aborting.")
            return None
        g = postgetter()
        g.verbose = True
        g.space = self
        g.struct = struct
        origin = [p() for p in self]
        try:
            m = scipy.optimize.fmin_powell(g, origin, full_output=True, xtol=xtol, ftol=ftol, maxiter=maxiter)
            ret = -0.5 * m[1] # log-likelihood for best point
        except:
            print("ParameterSpace.optimize: warning -- some kind of error in scipy.optimize.fmin_powell.")
            for i, p in enumerate(self):
                p.set(origin[i])
            ret = None
        for i, p in enumerate(self):
            p.set(m[0][i])
        #print m[2]
        return ret
        


class Updater:
    """
    Abstract base class for updaters. Do not instantiate directly.
    Constructor arguments:
     1* ParameterSpace to update.
     2* Step (proposal method) to use. This can be safely changed later using the set_step( ) method.
     3  Number of steps between proposal adaptations (zero to not adapt).
        For parallel chains, the Gelman-Rubin convergence criterion is calculated at the same interval.
     4  Number of steps before the first proposal adaptation.
     5  A function (with one arbitrary argument) to call after each adaptation.
     6  For no parallelization, set to None.
        For filesystem parallelization, set to a unique, scalar identifier.
        For MPI parallelization, set to an mpi4py.MPI.Comm object (e.g. MPI.COMM_WORLD)
        See module docstring for more details.
    """
    def __init__(self, space, step, every, start, on_adapt, parallel):
        self.space = space
        self.set_step(step)
        self.engine = None
        self.index = None
        self.adapt = (every > 0)
        self.adapt_every = every
        self.adapt_start = max(1, start)
        self.onAdapt = on_adapt
        self.count = 0
        if self.adapt:
            self.R = None
        self.rate = 1
    def restore(self, filename):
        f = open(filename, 'rb')
        s = pickle.load(f)
        self.restoreBits(s)
        f.close()
    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.saveBits(), f)
        f.close()
    def set_step(self, step):
        if step is None:
            return
        # todo: prevent direct assignment bypassing this
        self.step = step
        self.step.updater = self        

class CartesianUpdater(Updater):
    """
    Abstract base class for updaters that proposal one parameter at a time. Do not instantiate directly.
    """
    def __init__(self, space, step, adapt_every, adapt_starting, on_adapt, parallel):
        Updater.__init__(self, space, step, adapt_every, adapt_starting, on_adapt, parallel)
        if self.adapt:
            self.means = np.zeros(len(self.space))
            self.variances = np.zeros(len(self.space))
            if parallel is not None:
                mpi = False
                try:
                    if isinstance(parallel, MPI.Comm):
                        mpi = True
                        self.gatherAdapt = self.gatherMPI
                        self.comm = parallel
                except NameError:
                    pass
                if not mpi:
                    self.pid = str(parallel) 
                    self.uind = '_' + str(self.index)
                    self.gatherAdapt = self.gatherFilesys
            else:
                self.gatherAdapt = self.gatherSerial
        self.current_direction = 0
        self.origin = 0.0
    def __call__(self, struct):
        if self.adapt and self.count >= self.adapt_start and self.count % self.adapt_every == 0:
            self.do_adapt(struct)
        self.count += 1
        for j in range(len(self.space)):
            self.choose_direction(j)
            self.origin = self.space[self.current_direction]()
            self.step(struct)
            self.accumulate()
    def accumulate(self):
        if self.adapt:
            # Golub, Chan and Levesque one-pass mean and variance algorithm
            d = self.space[self.current_direction]() - self.means[self.current_direction]
            self.means[self.current_direction] += d / self.count
            # this is actually (n-1) times the variance (below)
            self.variances[self.current_direction] += (self.count-1.0)/self.count * d**2
    def do_adapt(self, struct):
        stdevs = self.gatherAdapt()
        for i, p in enumerate(self.space):
            if stdevs[i] != 0.0:
                p.width = stdevs[i]
        if not self.onAdapt is None:
            self.onAdapt(struct)
    def gatherFilesys(self):
        filename = parallel_filename_base + self.pid + self.uind + parallel_filename_ext
        try:
            self.save(filename)
        except:
            print("Warning: IO error while writing " + filename + " to store covariance (process " + self.pid + ", updater " + self.uind + ").")
        total = 0
        moment1 = np.zeros(len(self.space))
        moment2 = np.zeros(len(self.space))
        grandMeans = np.zeros(len(self.space))
        grandMeanVar = np.zeros(len(self.space))
        grandVarMean = np.zeros(len(self.space))
        j = 0
        for filename in glob.iglob(parallel_filename_base + '*' + self.uind + parallel_filename_ext):
            try:
                f = open(filename, 'rb')
                s = pickle.load(f)
                f.close()
                total += s['count'] # becomes Ntot
                moment1 += s['count'] * s['means'] # becomes Ntot*<x>
                moment2 += s['count'] * ( s['variances']/(s['count']-1.0) + s['means']**2 ) # becomes Ntot*<x^2>
                d = s['means'] - grandMeans
                grandMeans += d / (j+1.0)
                grandMeanVar += j/(j+1.0) * d**2
                d = s['variances']/(s['count']-1.0) - grandVarMean
                grandVarMean += d / (j+1.0)
                j += 1
            except:
                print("Warning: IO error while reading " + filename + " to update covariance (process " + self.pid + ", updater " + self.uind + ").")
        if j > 1:
            B = self.count / (j - 1.0) * grandMeanVar
            W = grandVarMean / j
            self.R = np.sqrt( (self.count-1.0)/self.count + B/(self.count*W) )
        return np.sqrt( (moment2 - moment1**2/total) / (total - 1.0) )
    def gatherMPI(self):
        alls = self.saveBits()
        alls = self.comm.allgather(alls)
        total = 0
        moment1 = np.zeros(len(self.space))
        moment2 = np.zeros(len(self.space))
        grandMeans = np.zeros(len(self.space))
        grandMeanVar = np.zeros(len(self.space))
        grandVarMean = np.zeros(len(self.space))
        for j, s in enumerate(alls):
            total += s['count'] # becomes Ntot
            moment1 += s['count'] * s['means'] # becomes Ntot*<x>
            moment2 += s['count'] * ( s['variances']/(s['count']-1.0) + s['means']**2 ) # becomes Ntot*<x^2>
            d = s['means'] - grandMeans
            grandMeans += d / (j+1.0)
            grandMeanVar += j/(j+1.0) * d**2
            d = s['variances']/(s['count']-1.0) - grandVarMean
            grandVarMean += d / (j+1.0)
        if len(alls) > 1:
            B = self.count / (len(alls) - 1.0) * grandMeanVar
            W = grandVarMean / len(alls)
            self.R = np.sqrt( (self.count-1.0)/self.count + B/(self.count*W) )
        return np.sqrt( (moment2 - moment1**2/total) / (total - 1.0) )
    def gatherSerial(self):
        return np.sqrt( self.variances / (self.count-1.0) )
    def move(self, x):
        p = self.space[self.current_direction]
        p.set(self.origin + x * p.width)
    def restoreBits(self, s):
        self.count = s['count']
        if s['type'] == 'Cartesian':
            for i, p in enumerate(self.space):
                p.width = s['widths'][i]
            self.means = s['means']
            self.variances = s['variances']
        elif  s['type'] == 'MultiDim':
            for i, p in enumerate(self.space):
                p.width = np.sqrt(s['covariances'][i,i])
            self.means = s['means']
            self.variances = s['covariances'].diagonal()
        else:
            raise Exception('CartesianUpdater.restoreBits: error restoring updater state -- unknown updater type')
    def saveBits(self):
        if self.adapt:
            return {'type': 'Cartesian', 'count': self.count, 'means': self.means, 'variances': self.variances, 'widths': [p.width for p in self.space]}
        else:
            return None
    def scatter(self, struct, ntries=10):
        c = self.current_direction
        origin = [p() for p in self.space]
        for i in range(ntries):
            for self.current_direction in range(len(self.space)):
                self.origin = origin[self.current_direction]
                self.move( np.random.randn() )
            self.engine.current_logP = self.space.log_posterior(struct)
            if self.engine.current_logP != -np.inf:
                self.current_direction = c
                return True
        for self.current_direction in range(len(self.space)):
            self.origin = origin[self.current_direction]
            self.move(0.0)
        self.engine.current_logP = self.space.log_posterior(struct)
        self.current_direction = c
        return False
    def set_covariance(self, cov):
        ok = True
        for i, p in enumerate(self.space):
            if cov[i,i] > 0:
                p.width = np.sqrt(cov[i][i])
            else:
                ok = False
        return ok
    def set_covariance_from_hessian(self, struct, h=0.1):
        ok = True
        g = postgetter()
        g.space = self.space
        g.struct = struct
        if self.engine.current_logP is None:
            self.engine.current_logP =  self.space.log_posterior(struct)
        chisq1 = -2.0 * self.engine.current_logP
        origin = [p() for p in self.space]
        for i, p in enumerate(self.space):
            trial = origin
            trial[i] = (1.0-h) * origin[i]
            chisq0 = g(trial)
            trial[i] = (1.0+h) * origin[i]
            chisq2 = g(trial)
            d2 = (chisq2 - 2.0*chisq1 + chisq0) / (h * origin[i])**2
            if d2 > 0.0:
                p.width = 1.0 / np.sqrt(d2)
            else:
                ok = False
        for i, p in enumerate(self.space):
            p.set(origin[i])
        return ok
        # if not have_numdifftools:
        #     print "Error: numdifftools package is required to calculate Hessian matrix"
        #     return False
        # origin = [p() for p in self.space]
        # g = postgetter()
        # g.space = self.space
        # g.struct = struct
        # try:
        #     Hfun = numdifftools.Hessdiag(g)
        #     m = Hfun(origin)
        #     good = True
        # except:
        #     print "CartesianUpdater.set_covariance_from_hessian: warning -- aborting due to Hessian evaluation failure"
        #     good = False
        # for i, p in enumerate(self.space):
        #     p.set(origin[i])
        # if not good:
        #     return False
        # for i, p in enumerate(self.space):
        #     if m[i] > 0.0:
        #         p.width = 1.0 / np.sqrt(m[i])
        #     else:
        #         good = False
        # return good


class SequentialUpdater(Updater):
    """
    Abstract class for updaters that propose each parameter in order.
    """
    def choose_direction(self, j):
        self.current_direction = j

class PermutationUpdater(Updater):
    """
    Abstract class for updaters that propose parameters in random order.
    """
    def __init__(self):
        self.permutation = np.arange(len(self.space))
        if self.count % len(self.space) != 0:
            np.random.shuffle(self.permutation)
    def choose_direction(self, j):
        if j == 0:
            np.random.shuffle(self.permutation)
        self.current_direction = self.permutation[j]

class CartesianSequentialUpdater(CartesianUpdater, SequentialUpdater):
    """
    Updater class to propose parameters individually, in sequence. See Updater.__init__.
    """
    def __init__(self, parameter_space, step, adapt_every=0, adapt_starting=0, on_adapt=None, parallel=None):
        CartesianUpdater.__init__(self, parameter_space, step, adapt_every, adapt_starting, on_adapt, parallel)

class CartesianPermutationUpdater(CartesianUpdater, PermutationUpdater):
    """
    Updater class to propose parameters individually, in random order. See Updater.__init__.
    """
    def __init__(self, parameter_space, step, adapt_every=0, adapt_starting=0, on_adapt=None, parallel=None):
        CartesianUpdater.__init__(self, parameter_space, step, adapt_every, adapt_starting, on_adapt, parallel)
        PermutationUpdater.__init__(self)



class MultiDimUpdater(Updater):
    """
    Abstract base class for block updates. Do not instantiate directly.
    """
    def __init__(self, space, step, adapt_every, adapt_starting, on_adapt, parallel):
        Updater.__init__(self, space, step, adapt_every, adapt_starting, on_adapt, parallel)
        self.rescale = 1.0
        if self.adapt:
            self.means = np.zeros(len(self.space))
            self.d = np.zeros(len(self.space))
            self.covariances = np.zeros((len(self.space), len(self.space)))
            if parallel is not None:
                mpi = False
                try:
                    if isinstance(parallel, MPI.Comm):
                        mpi = True
                        self.gatherAdapt = self.gatherMPI
                        self.comm = parallel
                except NameError:
                    pass
                if not mpi:
                    self.pid = str(parallel)
                    self.uind = '_' + str(self.index)
                    self.gatherAdapt = self.gatherFilesys
            else:
                self.gatherAdapt = self.gatherSerial
        self.current_direction = np.zeros(len(self.space))
        self.origin = np.zeros(len(self.space))
        self.widths = [p.width for p in self.space]
        self.width = 0.0
        self.basis = np.eye(len(self.space), len(self.space))
    def __call__(self, struct):
        if self.adapt and self.count >= self.adapt_start and self.count % self.adapt_every == 0:
            self.do_adapt(struct)
        self.choose_direction()
        self.origin = [p() for p in self.space]
        self.step(struct)
        self.accumulate()
    def accumulate(self):
        self.count += 1
        if self.adapt:
            for i, p in enumerate(self.space):
                self.d[i] = p() - self.means[i];
                self.means[i] += self.d[i] / self.count;
                for j in range(i+1):
                    self.covariances[i,j] += (self.count-1.0)/self.count * self.d[i] * self.d[j]
                    # don't need anything in the upper triangle
    def do_adapt(self, struct):
        cov = self.gatherAdapt()
        if self.set_covariance(cov) and not self.onAdapt is None:
            self.onAdapt(struct)
    def gatherFilesys(self):
        filename = parallel_filename_base + self.pid + self.uind + parallel_filename_ext
        try:
            self.save(filename)
        except:
            print("Warning: IO error while writing " + filename + " to store covariance (process " + self.pid + ", updater " + self.uind + ").")
        total = 0
        moment1 = np.zeros(len(self.space))
        moment2 = np.zeros( (len(self.space), len(self.space)) )
        grandMeans = np.zeros(len(self.space))
        grandMeanVar = np.zeros(len(self.space))
        grandVarMean = np.zeros(len(self.space))
        j = 0
        for filename in glob.iglob(parallel_filename_base + '*' + self.uind + parallel_filename_ext):
            try:
                f = open(filename, 'rb')
                s = pickle.load(f)
                f.close()
                total += s['count'] # becomes Ntot
                moment1 += s['count'] * s['means'] # becomes Ntot*<x>
                moment2 += s['count'] * ( s['covariances']/(s['count']-1.0) + np.outer(s['means'], s['means']) ) # becomes Ntot*<xy>
                d = s['means'] - grandMeans
                grandMeans += d / (j+1.0)
                grandMeanVar += j/(j+1.0) * d**2
                d = s['covariances'].diagonal()/(s['count']-1.0) - grandVarMean
                grandVarMean += d / (j+1.0)
                j += 1
            except:
                print("Warning: IO error while reading " + filename + " to update covariance (process " + self.pid + ", updater " + self.uind + ").")
        if j > 1:
            B = self.count / (j - 1.0) * grandMeanVar
            W = grandVarMean / j
            self.R = np.sqrt( (self.count-1.0)/self.count + B/(self.count*W) )
        return (moment2 - np.outer(moment1/total, moment1)) / (total - 1.0)
    def gatherMPI(self):
        alls = self.saveBits()
        alls = self.comm.allgather(alls)
        total = 0
        moment1 = np.zeros(len(self.space))
        moment2 = np.zeros( (len(self.space), len(self.space)) )
        grandMeans = np.zeros(len(self.space))
        grandMeanVar = np.zeros(len(self.space))
        grandVarMean = np.zeros(len(self.space))
        for j, s in enumerate(alls):
            total += s['count'] # becomes Ntot
            moment1 += s['count'] * s['means'] # becomes Ntot*<x>
            moment2 += s['count'] * ( s['covariances']/(s['count']-1.0) + np.outer(s['means'], s['means']) ) # becomes Ntot*<xy>
            d = s['means'] - grandMeans
            grandMeans += d / (j+1.0)
            grandMeanVar += j/(j+1.0) * d**2
            d = s['covariances'].diagonal()/(s['count']-1.0) - grandVarMean
            grandVarMean += d / (j+1.0)
        if len(alls) > 1:
            B = self.count / (len(alls) - 1.0) * grandMeanVar
            W = grandVarMean / len(alls)
            self.R = np.sqrt( (self.count-1.0)/self.count + B/(self.count*W) )
        return (moment2 - np.outer(moment1/total, moment1)) / (total - 1.0)
    def gatherSerial(self):
        return self.covariances / (self.count-1.0)
    def move(self, x):
        for i, p in enumerate(self.space):
            p.set(self.origin[i] + x * self.current_direction[i] * self.width)
    def restoreBits(self, s):
        self.count = s['count']
        if s['type'] == 'Cartesian':
            self.means = s['means']
            self.covariances = np.eye(len(self.space), len(self.space)) * s['variances']
            self.widths = s['widths']
            self.basis = np.eye(len(self.space), len(self.space))
        elif  s['type'] == 'MultiDim':
            self.means = s['means']
            self.covariances = s['covariances']
            self.widths = s['widths']
            self.basis = s['basis']
        else:
            raise Exception('MultiDimUpdater.restoreBits: error restoring updater state -- unknown updater type')
    def saveBits(self):
        if self.adapt:
            return {'type': 'MultiDim', 'count': self.count, 'means': self.means, 'covariances': self.covariances, 'widths': self.widths, 'basis': self.basis}
        else:
            return None
    def scatter(self, struct, ntries=10):
        for i in range(ntries):
            self.origin = [p() for p in self.space]
            for j in range(len(self.space)):
                self.current_direction = self.basis[:,j]
                self.width = self.widths[j]
                self.move( np.random.randn() )
            self.engine.current_logP = self.space.log_posterior(struct)
            if self.engine.current_logP != -np.inf:
                return True
        for j in range(len(self.space)):
            self.current_direction = self.basis[:,j]
            self.move(0.0)
            self.engine.current_logP = self.space.log_posterior(struct)
        return False
    def set_covariance(self, cov):
        try:
            evals, self.basis = np.linalg.eigh(cov)
            self.widths = [self.rescale*np.sqrt(abs(v)) for v in evals]
            return True
        except np.linalg.LinAlgError:
            print("MultiDimUpdater.set_covariance: warning -- aborting due to covariance diagonalization failure")
            return False
    def set_covariance_from_hessian(self, struct, h=0.1):
        ok = True
        g = postgetter()
        g.space = self.space
        g.struct = struct
        if self.engine.current_logP is None:
            self.engine.current_logP =  self.space.log_posterior(struct)
        chisq1 = -2.0 * self.engine.current_logP
        self.origin = [p() for p in self.space]
        self.basis = np.eye(len(self.space), len(self.space))
        for i, p in enumerate(self.space):
            trial = self.origin
            trial[i] = (1.0-h) * self.origin[i]
            chisq0 = g(trial)
            trial[i] = (1.0+h) * self.origin[i]
            chisq2 = g(trial)
            d2 = (chisq2 - 2.0*chisq1 + chisq0) / (h * self.origin[i])**2
            if d2 > 0.0:
                self.widths[i] = 1.0 / np.sqrt(d2)
            else:
                ok = False
        for i, p in enumerate(self.space):
            p.set(self.origin[i])
        return ok
        # if not have_numdifftools:
        #     print "Error: numdifftools package is required to calculate Hessian matrix"
        #     return False
        # self.origin = [p() for p in self.space]
        # g = postgetter()
        # g.verbose = True
        # g.space = self.space
        # g.struct = struct
        # try:
        #     Hfun = numdifftools.Hessian(g, numTerms=0, stepRatio=1.01) #stepNom=self.widths, 
        #     m = Hfun(self.origin)
        #     good = True
        # except:
        #     print "MultiDimUpdater.set_covariance_from_hessian: warning -- aborting due to Hessian evaluation failure"
        #     good = False
        # for i, p in enumerate(self.space):
        #     p.set(self.origin[i])
        # if not good:
        #     return False
        # try:
        #     cov = np.linalg.inv(m)
        #     return self.set_covariances(cov)
        # except np.linalg.LinAlgError:
        #     print "MultiDimUpdater.set_covariance_from_hessian: warning -- aborting due to Hessian inversion failure"
        #     return False
        
class MDSequentialUpdater(Updater):
    """
    Abstract class for sequential block updates.
    """
    def choose_direction(self):
        j = self.count % len(self.space)
        self.current_direction = self.basis[:,j]
        self.width = self.widths[j]

class MDPermutationUpdater(Updater):
    """
    Abstract class for block updates in random order.
    """
    def __init__(self):
        self.permutation = np.arange(len(self.space))
        if self.count % len(self.space) != 0:
            np.random.shuffle(self.permutation)
    def choose_direction(self):
        j = self.count % len(self.space)
        if j == 0:
            np.random.shuffle(self.permutation)
        self.current_direction = self.basis[:, self.permutation[j]]
        self.width = self.widths[self.permutation[j]]

class MDRotationUpdater(Updater):
    """
    Abstract class for block updates in random directions.
    """
    def __init__(self):
        self.j = 0
        self.q0 = 0
        self.q1 = 1
        self.cosr = 1
        self.sinr = 0
    def choose_direction(self):
        # Choose a random pair of basic vectors and rotate randomly in that plane.
        # On the next call, try the orthogonal direction.
        if self.j == 0:
            self.q0 = np.random.randint(0, len(self.space))
            self.q1 = np.random.randint(0, len(self.space)-1)
            if self.q0 == self.q1:
                self.q1 = len(self.space) - 1
            r = np.random.random() * 2.0 * np.pi
            self.cosr = np.cos(r)
            self.sinr = np.sin(r)
            self.current_direction = self.cosr * self.widths[self.q0] * self.basis[:,self.q0] - self.sinr * self.widths[self.q1] * self.basis[:,self.q1]
        else:
            self.current_direction = self.sinr * self.widths[self.q0] * self.basis[:,self.q0] + self.cosr * self.widths[self.q1] * self.basis[:,self.q1]
        self.width = np.sqrt( sum( self.current_direction**2 ) )
        self.current_direction /= self.width
        self.j = (self.j + 1) % 2
    # NB: the below seems more elegant, but appears to have a fatal bug.
#     def choose_direction(self):
#         self.current_direction = np.random.randn(len(self.space))
#         self.current_direction /= np.sqrt( sum( self.current_direction**2 ) )
#         self.width = np.sqrt( 1.0 / sum( (self.current_direction/self.widths)**2 ) )
#         self.current_direction = np.dot(self.basis, self.current_direction)

class MultiDimSequentialUpdater(MultiDimUpdater, MDSequentialUpdater):
    """
    Updater class to propose block-update parameters in sequence. See Updater.__init__.
    """
    def __init__(self, parameter_space, step, adapt_every=0, adapt_starting=0, on_adapt=None, parallel=None):
        MultiDimUpdater.__init__(self, parameter_space, step, adapt_every, adapt_starting, on_adapt, parallel)

class MultiDimPermutationUpdater(MultiDimUpdater, MDPermutationUpdater):
    """
    Updater class to block-update parameters in random order. See Updater.__init__.
    """
    def __init__(self, parameter_space, step, adapt_every=0, adapt_starting=0, on_adapt=None, parallel=None):
        MultiDimUpdater.__init__(self, parameter_space, step, adapt_every, adapt_starting, on_adapt, parallel)
        MDPermutationUpdater.__init__(self)

class MultiDimRotationUpdater(MultiDimUpdater, MDRotationUpdater):
    """
    Updater class to block-update parameters in random directions. See Updater.__init__.
    """
    def __init__(self, parameter_space, step, adapt_every=0, adapt_starting=0, on_adapt=None, parallel=None):
        MultiDimUpdater.__init__(self, parameter_space, step, adapt_every, adapt_starting, on_adapt, parallel)
        MDRotationUpdater.__init__(self)


class GWsqrtdist:
    """
    Functor for sampling from the stretch function remommended by Goodman and Weare for their sampling method.
    g(z) ~ 1/sqrt(z) if 1/a <= z <= a, otherwise 0
    The CDF is 2*(sqrt(z)-sqrt(1/a)) / (2*(sqrt(a)-sqrt(1/a))), whose inverse is ( u * 2*(sqrt(a)-sqrt(1/a)) / 2 + sqrt(1/a) )^2.
    """
    def __init__(self, a=2.0):
        self.mult = 2.0*(np.sqrt(a)-np.sqrt(1.0/a)) / 2.0
        self.add = np.sqrt(1.0/a)
    def __call__(self):
        return (np.random.random_sample() * self.mult + self.add)**2
class GoodmanWeareUpdater(Updater):
    """
    Implements the ensemble sampling algorithm of
    Goodman, J. & Weare, J., 2010, Comm. App. Math. Comp. Sci., 5, 65
    (as described in Foreman-Mackey et al. http://arxiv.org/abs/1202.3665).
    
    This does not generate a Markov chain as such, but rather evolves an ensemble of 'walker' points to approximate samples from the target distribution. The traditional Backend architecture doesn't make a lot of sense in this context, although the updater arranges to expose a different walker after each call. Instead, snapshots of the whole ensemble should be recorded periodically (e.g. through the Engine's on_step option). A TODO is to implement a more general EnsembleSampler class, with an appropriate interface to Backends. Parallelization is implemented via MPI (exclusively, at the moment).
    
    *IMPORTANT*: the walkers MUST not be started at exactly the same location. `scatter' will attempt to move them randomly, according to the `width' member of each Parameter, but you might want something more sophisticated.
    
    Class-specific constructor arguments:
    - nwalkers: a positive, even integer giving the size of the ensemble. Foreman-Mackey advocates making this as big as possible, subject to performace issues. At minimum, should be a couple of times the size of the parameter space.
    - stretch: a functor returning a sample from the 'stretch' distribution of the algorithm. Note that this cannot be an arbitrary function; see references above. The default is recommended by these references.
    - initialize: whether to set the position of every walker to the current values in space. See warning above.
    - comm: an MPI communicator, if using mpi4py. If not None, the walkers will be divided up, and the ensemble will be evolved according to the parallel algorithm given in Foreman-Mackey et al.
    """
    def __init__(self, space, nwalkers, stretch=GWsqrtdist(2.0), initialize=True, comm=None):
        if nwalkers <= 0 or nwalkers % 2 != 0:
            raise Exception('GoodmanWeareUpdater: number of walkers must be even and positive')
        self.space = space
        self.nwalkers = nwalkers
        self.comm = comm
        self.mywalkers = [range(nwalkers), []]
        self.complement = [ [0,nwalkers-1] ]
        self.stretch = stretch
        self.ensemble = np.ndarray([nwalkers,len(space)])
        self.lnP = [None] * nwalkers
        Updater.__init__(self, space, None, 0, 0, None, None)
        if (initialize):
            for j,p in enumerate(space):
                self.ensemble[:,j] = p()
        if comm is not None:
            commsize = comm.Get_size()
            rank = comm.Get_rank()
            assignments = np.array(range(commsize) * np.ceil(1.0*nwalkers/commsize))[0:nwalkers]
            mine = np.where(assignments == rank)[0]
            self.mywalkers[0] = mine[np.where(mine < nwalkers/2)]
            self.mywalkers[1] = mine[np.where(mine >= nwalkers/2)]
            self.complement = [ [nwalkers/2, nwalkers-1], [0, nwalkers/2-1] ]
    def __call__(self, struct):
        self.count += 1
        for half in [0,1]:
            for k in self.mywalkers[half]:
                # get lnP for initial position, if necessary
                if self.lnP[k] is None:
                    for j,p in enumerate(self.space):
                        p.set(self.ensemble[k,j])
                    self.lnP[k] = self.space.log_posterior(struct)
                # choose a different walker
                j = k
                while j == k: # check only needs to be done for non-parallel runs
                    j = np.random.random_integers(self.complement[half][0], self.complement[half][1]) # in [x1,x2]
                # propose a stretch move
                z = self.stretch()
                prop = self.ensemble[j,:] + z * (self.ensemble[k,:] - self.ensemble[j,:])
                # get the likelihood and accept/reject
                for j,p in enumerate(self.space):
                    p.set(prop[j])
                trial_logP = self.space.log_posterior(struct)
                lnq = (len(self.space) - 1.0) * np.log(z) + trial_logP - self.lnP[k]
                lnr = np.log(np.random.random_sample())
                if lnr <= lnq:
                    self.ensemble[k,:] = prop
                    self.lnP[k] = trial_logP
            if self.comm is not None:
                self.broadcast(half)
        # set "current" parameter values to one of the walkers (for Backends to write)
        k = self.count % len(self.nwalkers)
        for j,p in enumerate(self.space):
            p.set(self.ensemble[k,j])
    def broadcast(self, half):
        alls = {'indices':self.mywalkers[half], 'values':self.ensemble[self.mywalkers[half],:]}
        alls = self.comm.allgather(alls)
        for j,s in enumerate(alls):
            self.ensemble[s['indices'],:] = s['values']
    def scatter(self, struct, ntries=10):
        allgood = True
        for half in [0,1]:
            for k in self.mywalkers[half]:
                good = False
                for i in range(ntries):
                    prop = self.ensemble[k,:] + np.random.randn(len(self.space)) * np.array([p.width for p in self.space])
                    for j,p in enumerate(self.space):
                        p.set(prop[j])
                    lnP = self.space.log_posterior(struct)
                    if lnP != -np.inf:
                        self.ensemble[k,:] = prop
                        self.lnP[k] = lnP
                        good = True
                        break
                allgood = allgood and good
        if self.comm is not None:
            self.broadcast(0)
            self.broadcast(1)
        return allgood
    def restoreBits(self, s):
        if s['type'] == 'GoodmanWeare':
            self.ensemble = s['ensemble']
            self.lnP = s['lnP']
        else:
            raise Exception('emceeUpdater.restoreBits: incompatible updater type')
    def saveBits(self):
        return {'type':'GoodmanWeare' , 'ensemble': self.ensemble, 'lnP': self.lnP}

class TableUpdater(Updater):
    """
    Updater which can only propose updates from a tabulated set of parameter values, assumed to have equal proposal density. The only possible steppers in this case are Metropolis-Hastings and Gibbs; Metropolis is used here. The table passed to the constructor should be a numpy record array with column names corresponding to the names of the parameters in space. No adaptation is implemented, although in principle something could be done.
    The constructor takes a ParameterSpace and a list of parameter names (corresponding to the columns of the table). It will search the space for those parameters and associate itself with them, AND WILL REMOVE THEM FROM SPACE, since in general they should not be varied by some other updater.
    Note that this updater really only restricts the allowed values of the appropriate parameters to the given sets of values. It's up to you to associate a term in the log-likelihood with the row number, if appropriate.
    """
    def __init__(self, space, table, names=None):
        step = Metropolis(proposal_length=lambda:1.0)
        self.table = table
        self.nchoices = table.shape[0]
        self.names = table.dtype.names
        if names is not None:
            self.names = names
        Updater.__init__(self, space, step, 0, 0, None, None)
        self.pdict = {}
        sp = list(space)
        for p in sp:
            #print p.name
            for i,n in enumerate(self.names):
                if n == p.name:
                    #print "match", p.name, n, i
                    self.pdict[p.name] = (i,p)
                    space.remove(p)
                    break
        self.move(1.0) # start at some random point
        self.last = self.j
    def __call__(self, struct):
        self.count += 1
        self.step(struct)
        self.last = self.j
    def move(self, x):
        if x == 0.0: # failed metropolis step
            self.j = self.last
        else:
            self.j = np.random.random_integers(self.nchoices) - 1
        for (n,pp) in self.pdict.items():
            pp[1].set( self.table[self.j][pp[0]] )
    def restoreBits(self, s):
        self.count = s['count']
        if s['type'] == 'Table':
            self.table = s['table']
            self.nchoices = s['nchoices']
            self.j = s['j']
            self.last = s['last']
            self.names = s['names']
        else:
            raise Exception('TableUpdater.restoreBits: error restoring updater state -- wrong updater type')
    def saveBits(self):
        return {'type': 'Table', 'table': self.table, 'nchoices': self.nchoices, 'j': self.j, 'last': self.last, 'names': self.names}
    def scatter(self, struct, ntries=10):
        for i in range(ntries):
            self.move(1.0)
            self.engine.current_logP = self.space.log_posterior(struct)
            if self.engine.current_logP != -np.inf:
                return True
        return False
    def set_covariance(self, cov):
        return True
    def set_covariance_from_hessian(self, struct, h=0.1):
        return True
    
class Step:
    """
    Abstract base class for proposal methods. Do not instantiate directly.
    """
    def __init__(self):
        self.updater = None

class Slice(Step):
    """
    Class implementing the slice proposal method (http://arxiv.org/abs/physics/0009028).
    Constructor arguments:
     1. factor by which to increase the initial slice width.
     2. maximum number of iterations in stepping-out and stepping-in loops before giving up.
     3. whether to suppress warnings if said loops reach the maximum number of iterations.
     4. whether to print a ridiculous amount of information (possibly useful for debugging posterior functions).
    """
    def __init__(self, width_factor=2.4, maxiter=100, quiet=True, obnoxious=False):
        self.width_fac = width_factor
        self.maxiter = maxiter
        self.quiet = quiet
        self.obnoxious = obnoxious
        Step.__init__(self)
    def __call__(self, struct):
        if self.updater.engine.current_logP is None:
            self.updater.engine.current_logP = self.updater.space.log_posterior(struct)
        z = self.updater.engine.current_logP - np.random.exponential() # log level of slice
        L = -self.width_fac * np.random.random_sample()                # left edge of the slice
        R = L + self.width_fac                                         # right edge
        if self.obnoxious:
            print('Slice: starting params:', [p() for p in self.updater.space])
            print('Slice: current level', self.updater.engine.current_logP, '; seeking', z)
            print('Slice: stepping out left')
        for i in range(self.maxiter):
            self.updater.move(L)
            lnew = self.updater.space.log_posterior(struct)
            if self.obnoxious:
                print('Slice: params:', [p() for p in self.updater.space])
                print('Slice:', L, lnew)
            if lnew <= z:
                break
            L -= self.width_fac;
        else:
            if not self.quiet:
                print("Slice(): warning -- exhausted stepping out (left) loop")
        if self.obnoxious:
            print('Slice: params:', [p() for p in self.updater.space])
            print('Slice: stepping out right')
        for i in range(self.maxiter):
            self.updater.move(R)
            lnew = self.updater.space.log_posterior(struct)
            if self.obnoxious:
                print('Slice: params:', [p() for p in self.updater.space])
                print('Slice:', R, lnew)
            if lnew <= z:
                break
            R += self.width_fac;
        else:
            if not self.quiet:
                print("Slice(): warning -- exhausted stepping out (right) loop")
        if self.obnoxious:
            print('Slice: stepping in')
        for i in range(self.maxiter):
            x1 = L + (R - L) *  np.random.random_sample()
            self.updater.move(x1)
            self.updater.engine.current_logP = self.updater.space.log_posterior(struct)
            if self.obnoxious:
                print('Slice: params:', [p() for p in self.updater.space])
                print('Slice:', x1, self.updater.engine.current_logP)
            if self.updater.engine.current_logP < z:
                if x1 < 0:
                    L = x1
                else:
                    R = x1
            else:
                break
        else:
            if not self.quiet:
                print("Slice(): warning -- exhausted stepping in loop")
        if self.obnoxious:
            print('Slice: completed')

class Metropolis(Step):
    """
    Class implementing the Metropolis (*not* Metropolis-Hastings) proposal algorithm.
    Constructor arguments:
     1. a function or functor (with zero arguments) returning a random proposal distance in units of the current estimated posterior width. Positive and negative numbers must be returned with equal probability. The default is simply a unit Gaussian random number.
    """
    def __init__(self, proposal_length=np.random.randn, width_factor=2.4):
        self.length = proposal_length
        self.width_fac = width_factor
        self.multiplicity = 0
        Step.__init__(self)
    def __call__(self, struct):
        if self.updater.engine.current_logP is None:
            self.updater.engine.current_logP = self.updater.space.log_posterior(struct)
        self.updater.move( self.width_fac * self.length() )
        trial_logP = self.updater.space.log_posterior(struct)
        delta_logP = trial_logP - self.updater.engine.current_logP
        r = np.log(np.random.random_sample())
        if delta_logP > 0.0 or r < delta_logP:
            self.updater.engine.current_logP = trial_logP
            self.multiplicity = 1
        else:
            self.updater.move(0.0)
            self.multiplicity += 1


class randNormalExp:
    """
    Functor for providing heavy-tailed proposal lengths: Exponential with probability <ratio> and Gaussian with probability 1-<ratio>.
    Constructor arguments: ratio.
    """
    def __init__(self, ratio=0.333333333):
        self.ratio = ratio
    def __call__(self):
        r = np.random.randn()
        if r <= self.ratio:
            if r < 0.5*self.ratio:
                return np.random.exponential()
            else:
                return -np.random.exponential()
        else:
            return np.random.randn()

class randChiExp:
    """
    Functor for providing heavy-tailed proposal lengths: Exponential with probability <ratio> and Chi(dof)/sqrt(dof) with probability 1-<ratio>. randChiExp(1/3, 2) is the CosmoMC default.
    Constructor arguments:
     1. ratio
     2. degrees of freedom
    """
    def __init__(self, ratio=0.3333333333, dof=2):
        self.ratio = ratio
        self.dof = 2
    def __call__(self):
        r = np.random.randn()
        if r <= self.ratio:
            if r < 0.5*self.ratio:
                return np.random.exponential()
            else:
                return -np.random.exponential()
        else:
            if r < 0.5*(1.0 + self.ratio):
                return np.sqrt( np.random.chisquare(self.dof)/self.dof )
            else:
                return -np.sqrt( np.random.chisquare(self.dof)/self.dof )


# not really necessary to inherit like this, but what the heck
class Backend:
    """
    Abstract base class for chain storage. Do not instantiate directly.
    """
    def __call__(self, space):
        pass

class textBackend(Backend):
    """
    Class to store a chain in a text file.
    Constructor argument: an open Python file object.
    Static function readtoDict( ) loads a chain from such a file into a dictionary.
    """
    def __init__(self, file, every=1, flush=False):
        self.file = file
        self.every = every
        self.flush = flush
        self.stored = []
        self.count = 0
    def __call__(self, space):
        self.count += 1
        st=''
        for p in space:
            st = st + ' ' + str(p())
        st = st
        self.stored.append(st)
        if self.count % self.every == 0:
            self.file.write( '\n'.join(self.stored) + '\n' )
            if self.flush:
                self.file.flush()
            self.stored = []
    def close(self):
        self.file.close()
    @classmethod
    def readToDict(cls, filename, quiet=True):
        d = None
        f = open(filename, 'r')
        line = f.readline()
        if line != '':
            d = {}
            keys = line.split()
            try:
                values = [float(key) for key in keys]
                for i, val in enumerate(values):
                    keys[i] = 'V' + str(i+1)
                    d[keys[i]] = [ val ]
            except ValueError:
                for key in keys:
                    d[key] = []
            while True:
                line = f.readline()
                if line == '':
                    break
                try:
                    values = [float(word) for word in line.split()]
                    for i, key in enumerate(keys):
                        d[key].append(values[i])
                except ValueError:
                    if not quiet:
                        print("textBackend.readToDict: ignoring line " + line)
        f.close()
        return d

class headerTextBackend(Backend):
    """
    Like textBackend, but automatically reads/writes a header line with the parameter names.
    """
    def __init__(self, file, space):
        self.fields = [p.name for p in space]
        self.writer = csv.DictWriter(file, self.fields, delimiter = ' ', quoting=csv.QUOTE_MINIMAL, )
        if sys.version_info < (2,7):
            self.writer.writer.writerow(self.writer.fieldnames)
        else:
            self.writer.writeheader()
    def __call__(self, space):
        towrite = {}
        for p in space:
            towrite[p.name] = p()
        self.writer.writerow(towrite)
    @classmethod
    def readToDict(cls, filename, quiet=True):
        db = {}
        reader = csv.DictReader(open(filename), delimiter = ' ', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            for key in row.keys():
                if key not in db:
                    db[key] = []
                db[key].append(float(row[key]))
        for key in db.keys():
            db[key] = np.array(db[key])
        return db

class stdoutBackend(textBackend):
    """
    Class to simply print a chain to the terminal without storing it.
    """
    def __init__(self):
        textBackend.__init__(self, sys.stdout)

class dictBackend(dict, Backend):
    """
    Class to store a chain in a dictionary (inherits dict). If a Parameter has a non-empty string-type name attribute, the corresponding key is that name, otherise it is a reference to the Parameter object itself.
    """
    def __call__(self, space):
        for p in space:
            key = p
            try:
                if p.name != '':
                    key = p.name
            except:
                pass
            try:
                self[key].append(p())
            except KeyError:
                self[key] = []
                self[key].append(p())

class binaryBackend(Backend):
    """
    Class to store a chain as binary data.
    Constructor argument: an open Python file object, wb or ab mode.
    """
    def __init__(self, file, format='f', every=1, flush=False):
        self.file = file
        self.format = format
        self.every = every
        self.flush = flush
        self.stored = []
        self.count = 0
        self.struct = None
    def __call__(self, space):
        self.count += 1
        if self.struct is None:
            self.struct = struct.Struct(str(len(space)) + self.format)
        ps = [p() for p in space]
        self.stored.append(ps)
        if self.count % self.every == 0:
            for ps in self.stored:
                self.file.write( self.struct.pack(*ps) )
            if self.flush:
                self.file.flush()
            self.stored = []
    def close(self):
        self.file.close()


class Engine(list):
    """
    Class to organize Updaters of ParameterSpaces and run the MCMC (inherits list).
    Constructor arguments:
     1. sequence of Updater objects. If Updaters are added any other way, the register_updater( ) method must be used.
     2. a ParameterSpace of Parameters whose values are to be stored at each step. This need not be the same as the ParameterSpace(s) referred to by the Updaters.
     3. a function of one argument to be called after each step (i.e. each time that each Updater has been called).
    To run a chain, use the () method. Arguments:
     1. number of iterations (every Updater is called for a single iteration).
     2. an object that is passed to the log_posterior, Updater.on_adapt, and on_step functions.
     3. a sequence of Backend objects where the chain is to be stored.
    """
    # todo: make sure directly assigned Updaters get registered
    def __init__(self, updaterList=[], parameterspace_to_track=None, on_step=None):
        list.__init__(self, updaterList)
        for i, updater in enumerate(self):
            self.register_updater(updater, i)
        self.space = parameterspace_to_track
        self.onStep = on_step
        self.count = 0
        self.current_logP = None
    def __setitem__(self, key, value):
        self[key] = value
        self.register_updater(value, key)
    def __call__(self, number=1, struct=None, backends=[stdoutBackend()]):
        try:
            for i in range(number):
                for updater in self:
                    for j in range(updater.rate):
                        updater(struct)
                self.count += 1
                if not self.onStep is None:
                    self.onStep(struct)
                if not self.space is None:
                    for backend in backends:
                        backend(self.space)
        except KeyboardInterrupt:
            print("Interrupted by keyboard with count = " + str(self.count))
    def append(self, v):
        list.append(self, v)
        self.register_updater(v, len(self))
    def register_updater(self, updater, index):
        updater.engine = self
        updater.index = index
        updater.uind = '_' + str(index)



class Vehicle:
    """
    A class to handle most setup steps in the common case where we want to use a single Updater for a single ParameterSpace.
    Constructor arguments:
     1. A ParameterSpace object for the analysis.
     2. A single Backend object for storing the chain.
     3. An Updater class. Note than this implementation is currently sub-optimal for GoodmanWeare.
     4. A Step class to use if appropriate for the chosen Updater.
     5. Whether to include log posterior values with the saved samples.
     6. Parallelization option, as in Updater.
     7. on_step option, as in Engine.
     8-10. Adaptation options, as in Updater.
     11. Filename to save checkpoints (Updater states) to after each adaptation.
     12-13. Options specific to GoodmanWeareUpdater.
    Run a chain by calling the object (see __call__ docstring).
    """
    def __init__(self, parameterSpace, backend=stdoutBackend(), updaterType=MultiDimRotationUpdater, updaterStep=Slice(), trace_lnP=True, parallel=None, on_step=None, adapt_every=None, adapt_starting=None, on_adapt=None, checkpoint=None, GW_nwalkers=None, GW_stretch=GWsqrtdist(2.0)):
        if adapt_every is None:
            adapt_every = max(100,10*len(parameterSpace))
        if adapt_starting is None:
            adapt_starting = adapt_every
        if GW_nwalkers is None:
            GW_nwalkers = 2*len(parameterSpace)
        self.default_R_check_interval = adapt_every
        self.cust_onStep = on_step
        self.cust_onAdapt = on_adapt
        self.backend = backend
        if len(parameterSpace) == 1:
            updaterType = CartesianSequentialUpdater
        # For now, treat GoodmanWeare as a special case, and assume that any parallelization is through MPI.
        # TODO: writing samples to the backend should also be done differently.
        if updaterType == GoodmanWeareUpdater:
            self.updater = GoodmanWeareUpdater(parameterSpace, GW_nwalkers, GW_stretch=GWsqrtdist(2.0), initialize=True, comm=parallel)
            trace_lnP = False # makes no sense in this case
        else:
            self.updater = updaterType(parameterSpace, updaterStep, adapt_every, adapt_starting, self._onAdapt, parallel)
        self.trace = ParameterSpace(parameterSpace)
        if trace_lnP:
            self.lnP = DerivedParameter(name='_LOGP_')
            self.trace.append(self.lnP)
        else:
            self.lnP = None
        self.engine = Engine([self.updater], self.trace, on_step=self._onStep)
        self.checkpoint = checkpoint
        try:
            self.updater.restore(checkpoint)
            print('Loaded checkpoint', checkpoint)
        except:
            pass
        # TODO: get last position from backend. This requires some specialization both for the backend type and the updater type, however.
    def _onAdapt(self, struct):
        if self.checkpoint is not None:
            self.updater.save(self.checkpoint)
        if self.cust_onAdapt is not None:
            self.cust_onAdapt(struct)
    def _onStep(self, struct):
        if self.lnP is not None:
            self.lnP.value = self.engine.current_logP
        if self.cust_onStep is not None:
            self.cust_onStep(struct)
    def __call__(self, minIter, maxIter, struct=None, scatter=True, R_target=1.1, R_check_interval=None):
        """
        Arguments:
         1. Minimum number of steps. If adapting and checking convergence, it's helpful to make this at least 1 larger than the adaptation interval. (This is because proposals are adapted before stepping, not after, so we need an extra step here in order for convergence values to be current later on.)
         2. Maximum number of steps (but not precisely; the chain will be run in increments of R_check_interval).
         3. The object to be passed to the posterior function.
         4. Whether to call the Updater's scatter method before starting.
         5. Target value of the Gelman-Rubin convergence criterion (or None). This is checked for every R_check_interval steps after the first minIter, and the chain will terminate before maxIter if the target is reached.
         6. Interval for checking convergence. Defaults to the interval for adaptation, since the convergence is only calculated in that step anyway.
        """
        if scatter and not self.updater.scatter(struct):
            raise Exception('Failed to find an acceptable starting position.')
        if R_check_interval is None:
            R_check_interval = self.default_R_check_interval
        self.engine(minIter, struct, [self.backend])
        niter = minIter
        while niter < maxIter:
            if self.updater.R is None or len(self.updater.R) == 1:
                R = self.updater.R
            else:
                R = max(self.updater.R)
            if R is not None:
                print('max R =', R)
                if R < R_target:
                    break
            self.engine(R_check_interval, struct, [self.backend])
            niter += R_check_interval
        
        
        
        

def example(number=None):
    if number == 1:
        print("""
# Here is a simple example. As shown it will run in non-parallel mode; comments indicate what to do for parallelization.

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
### E.g., usually it would also contain or point to the data being used to constrain the model. A good idea is to write the state of any updaters to a file after each adaptation (using the on_adapt functionality), in which case keeping pointers to the updaters here is convenient. Also commonly useful: a DerivedParameter which holds the value of the posterior log-density for each sample.
class Thing:
    def __init__(self, x, y):
        self.x = x
        self.y = y
thing = Thing(x, y)

### The log-posterior function. Here we just assume a bivariate Gaussian posterior with marginal standard deviations s(x)=2 and s(y)=3, correlation coefficient 0.75, and means <x>=-1, <y>=1.
def post(thing):
    r = 0.75
    sx = 2.0
    sy = 3.0
    mx = -1.0
    my = 1.0
    return -0.5/(1.0-r**2)*( (thing.x()-mx)**2/sx**2 + (thing.y()-my)**2/sy**2 - 2.0*r*(thing.x()-mx)/sx*(thing.y()-my)/sy )

### Use the Vehicle class to handle lots of setup for us. Accept all of the default behavior.
v = Vehicle(ParameterSpace([x,y], post))
### The first argument here (minimum number of iterations) is meaningless, since we need parallelization in order to stop before the maximum number of iterations (2nd argument).
v(100, 10000, thing)

## Alternative for using MPI, for e.g.:
#v = Vehicle(ParameterSpace([x,y], post), backend=textBackend(open("chain"+str(mpi_rank+1)+".txt",'a')), parallel=mpi_comm, checkpoint="chain"+str(mpi_rank+1)+".chk")
## In this case, convergence will be tested periodically, and the chains are allowed to terminate any time after the first 501 iterations if a default convergence criterion is satisfied.
#v(501, 10000, thing)
""")

    elif number == 2:
        print("""
# Here is a simple example. As shown it will run in non-parallel mode; comments indicate what to do for parallelization.

from lmc import *
## for MPI
#from mpi4py import MPI
#mpi_rank = MPI.COMM_WORLD.Get_rank()

### Define some parameters.
x = Parameter(name='x')
y = Parameter(name='y')

### This is the object that will be passed to the likelihood function.
### In this simple case, it just holds the parameter objects, but in general it could be anything.
### E.g., usually it would also contain or point to the data being used to constrain the model. A good idea is to write the state of any updaters to a file after each adaptation (using the on_adapt functionality), in which case keeping pointers to the updaters here is convenient. Also commonly useful: a DerivedParameter which holds the value of the posterior log-density for each sample.
class Thing:
    def __init__(self, x, y):
        self.x = x
        self.y = y
thing = Thing(x, y)

### The log-posterior function. Here we just assume a bivariate Gaussian posterior with marginal standard deviations s(x)=2 and s(y)=3, correlation coefficient 0.75, and means <x>=-1, <y>=1.
def post(thing):
    r = 0.75
    sx = 2.0
    sy = 3.0
    mx = -1.0
    my = 1.0
    return -0.5/(1.0-r**2)*( (thing.x()-mx)**2/sx**2 + (thing.y()-my)**2/sy**2 - 2.0*r*(thing.x()-mx)/sx*(thing.y()-my)/sy )

### Create a parameter space consisting of x and y, and associate the log-posterior function with it.
space = ParameterSpace([thing.x, thing.y], post)

### If we'd bothered to define a DerivedParameter in Thing which would hold the posterior density, we might want to define a larger ParameterSpace and pass it to the Engine later on to be saved in the Backends (instead of space).
#trace = ParameterSpace([thing.x, thing.y, thing.logP])

### Use slice sampling for robustness. Adapt the proposal distribution every 100 iterations starting with the 100th.
step = Slice()
parallel = None
## for MPI parallelization
# parallel = MPI.COMM_WORLD
## for parallelization via the filesystem, this would have to be set to a different value for each concurrently running instance
#parallel = 1
updater = MultiDimSequentialUpdater(space, step, 100, 100, parallel=parallel)

### Create an Engine and tell it to drive this Updater and to store the values of the free parameters.
engine = Engine([updater], space)

### Store the chain in a text file.
chainfile = open("chain.txt", 'w')
## For filesystem parallelization, each instance should write to a different file.
## For MPI, the same is true, e.g.
#chainfile = open("chain" + str(MPI.COMM_WORLD.Get_rank()) + ".txt", 'w')
backends = [ textBackend(chainfile) ]

### Print the chain to the terminal as well
backends.append( stdoutBackend() )

### Run the chain for 10000 iterations
engine(10000, thing, backends)

### Close the text file to clean up.
chainfile.close()

## If this was a parallel run, print the convergence criterion for each parameter.
# print updater.R
""")

    elif number == 3:
        print("""
Here I test out LMC on a simple problem: fitting a linear model with Gaussian intrinsic scatter to data (which we'll simulate).

A Jupyter notebook showing (almost) this example with plots and results can be found at
https://github.com/abmantz/lmc/blob/master/examples/line.ipynb

import numpy as np
import lmc

# simulate some data
true_alpha = 25.0
true_beta = 0.5
true_sigma = 10.0

xs = 100.0 * np.random.random(20)
ys = np.random.normal(true_alpha + true_beta*xs, true_sigma)

# Define separate functions for the prior, likelihood
# and posterior, even though it isn't strictly necessary.

# I'll be lazy and define the parameter objects at global scope.
# (The actual declarations of alpha, beta and sigma will come later.)
# In a complex fitting code, this would be very bad practice, but
# for simple scripts like this it should be fine.
# Similarly, I will just use the global `xs` and `ys` data.
# Consequently, these functions do not need arguments!

def lnPrior():
    if sigma() <= 0.0:
        return -np.inf
    # NB "uniform in angle" slope prior as in the example we're following
    # Jeffreys prior for the scatter
    return -1.5*np.log(1 + beta()** 2) - np.log(sigma())

def lnLikelihood():
    ymod = alpha() + beta()*xs
    return np.sum( -0.5*((ys - ymod) / sigma())**2  - np.log(sigma()) )

# A quirk of LMC is that the log_posterior function must have 1 argument
# but, given the choices above, we won't actually use it.
def lnPosterior(junk=None):
    return lnPrior() + lnLikelihood()

# Here we define Parameter objects for each free parameter, and bundle them
# into a ParameterSpace that knows about the posterior function.
# Each Parameter is given a starting value and a guess at the appropriate step size.
alpha = lmc.Parameter(25.0, 1.0, 'alpha')
beta = lmc.Parameter(0.5, 0.05, 'beta')
sigma = lmc.Parameter(10.0, 1.0, 'sigma')
space = lmc.ParameterSpace([alpha, beta, sigma], lnPosterior)

# Run a few chain and store it in memory.
# I'll be lazy again and use Vehicle with all default settings.
v = lmc.Vehicle(space, lmc.dictBackend())
Nsteps = 10000 # or whatever you want
v(1, Nsteps)
""")

    else:
        print("""
Usage: example(N), where N is one of:
 1. A very simple example where the posterior is bivariate Gaussian, using the Vehicle wrapper to handle setup.
 2. Same as (1), but with the hood open.
 3. Fitting a line plus Gaussian scatter.
""")




class ChiSquareLikelihood:
    """
    A class to simplify fitting models to Gaussian data. Assign a function to the 'priors' attribute to include non-(improper uniform) priors.
    """
    def __init__(self, model, y, err=None, x=None):
        self.model = model
        self.y = y
        if err is None:
            self.err = np.ones(len(y))
        else:
            self.err = err
        if x is None:
            self.x = np.zeros(len(y))
        else:
            self.x = x
        self.chisquare = DerivedParameter(name='ChiSquare')
    def __call__(self, struct):
        try:
            chisq = -2.0 * self.priors(struct)
        except AttributeError:
            chisq = 0.0
        for j, y in enumerate(self.y):
            chisq += ( (self.model(self.x[j], struct) - y)/self.err[j] )**2
        self.chisquare.value = chisq
        return -0.5 * chisq




# Todo:
# 1. An Updater class that simply goes through an existing sequence, for importance sampling.

