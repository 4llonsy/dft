# dft
My implementation of dft on a single atom using LDA, VWN exchange-correlation functional.
LSD yet to be implemented.

# Dependencies
numpy, sympy, scipy, concurrent, futures

# How to run?
Self explanatory, Run the main.py file. 

# Note
The program uses n-1 cores, when n cores are available for faster calculation of exchange-correlation potential. This can be edited in the xc_parallel_compute.py file.
