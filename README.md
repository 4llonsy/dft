# dft
My implementation of dft on a single atom using LDA, VWN exchange-correlation functional.
LSD yet to be implemented.

# Dependencies
numpy, sympy, scipy, concurrent, futures

# How to run?
Self explanatory, Run the main.py file. 

# Note
The program uses n-1 cores, when n cores are available for faster calculation of exchange-correlation potential. This can be edited in the xc_parallel_compute.py file.

# Benchmark
Atomic Reference Data for Electronic Structure Calculations, Atomic Total Energies and Eigenvalues
https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-7

# Reference
https://doi.org/10.1103/PhysRevA.55.191
