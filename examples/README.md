# LMC example usage

Files in this folder either reproduce (`.py`) or illustrate with a bit more discussion/plots (`.ipynb`) the examples that can be found in the `example()` code within LMC.

1. [Sampling a Gaussian I](simple_hoodClosed.py) (uses the `Vehicle` class; chain goes to `stdout`; demonstrates MPI parallelization)
2. [Sampling a Gaussian I (MPI enabled)](mpi_hoodClosed.py) (same as above, but with the MPI lines active instead of commented)
3. [Sampling a Gaussian II](simple_hoodOpen.py) (without using the `Vehicle` class; chain goes to a text file; demonstrates MPI parallelization)
4. [Fitting a line + scatter](line.ipynb) (chains stored in Python dictionaries)
