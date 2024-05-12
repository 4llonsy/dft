from dft import dft_solver
ec = {"1s": "1 1"}

"""
"1s": "1 0" denotes 1 electron in upspin and 1 in downspin of 1s orbital
ec = electronic configuration
N = number of points in the radial grid
Z = atomic number
beta = mixing parameter to mix old solution in the new solution
xc = exchange correlation functional
r_min = minimum value of r in the radial grid
r_max = minimum value of r in the radial grid
tol = tolerance value in energies
"""

if __name__=="__main__":
    He = dft_solver(ec, N = 1000, Z=2, beta= 1/2, xc="LDA-VWN", r_min=1e-6, r_max=50, tol=1e-6)
    He.solve(20)