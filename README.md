# cloud_computing
This project aims to implement a distributed algorithm, either by utilizing GPU computing, employing Map/Reduce, or by implementing the distribution of computations across multiple (virtual) machines yourself using primitives such as MPI.

We choose to implement a parallel version of Sequential Monte Carlo using MPI4py. This version was designed to work on multiple CPUs. This project follows up a work I have done with Nicolas Chopin and its package "particles". The previous works had nothing to do with parallelisation but gave me the idea to implement that.

You can run the distributing computation using "mpiexec -n number_of_processes python cloud_computing.py

