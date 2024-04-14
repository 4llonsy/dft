"""Code is faster when more cores are available. It uses n-1 cores when n cores are available."""

import sys
from dft import dft_solver
sys.stdout = sys.stderr

def name(obj):
    for name, value in globals().items():
        if id(value) == id(obj):
            return name

atoms = {
    "H": {"1s": "1 0"},
    "He": {"1s": "1 1"},
    "Li": {"1s": "1 1", "2s": "1 0"},
    "Be": {"1s": "1 1", "2s": "1 1"},
    "B": {"1s": "1 1", "2s": "1 1", "2p": "1 0"},
    "C": {"1s": "1 1", "2s": "1 1", "2p": "2 0"},
    "N": {"1s": "1 1", "2s": "1 1", "2p": "3 0"},
    "O": {"1s": "1 1", "2s": "1 1", "2p": "3 1"},
    "F": {"1s": "1 1", "2s": "1 1", "2p": "3 2"},
    "Ne": {"1s": "1 1", "2s": "1 1", "2p": "3 3"},
    "Na": {"1s": "1 1", "2s": "1 1", "2p": "3 3", "3s": "1 0"}, 
    "Mg": {"1s": "1 1", "2s": "1 1", "2p": "3 3", "3s": "1 1"}
}

#Calculating total energies of the first 12 elements as neutral atoms, and calculating total energies of singly positive ions
if __name__=="__main__":
    H = dft_solver(atoms["H"], N = 15788, Z=1, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    He = dft_solver(atoms["He"], N = 15788, Z=2, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    Li = dft_solver(atoms["Li"], N = 15788, Z=3, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    Be = dft_solver(atoms["Be"], N = 15788, Z=4, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    B = dft_solver(atoms["B"], N = 15788, Z=5, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    C = dft_solver(atoms["C"], N = 15788, Z=6, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    N = dft_solver(atoms["N"], N = 15788, Z=7, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    O = dft_solver(atoms["O"], N = 15788, Z=8, beta= 1/2, xc="LDA", r_min=1e-6, r_max=45, tol=1e-6)
    F = dft_solver(atoms["F"], N = 15788, Z=9, beta= 1/2, xc="LDA", r_min=1e-6, r_max=40, tol=1e-6)
    Ne = dft_solver(atoms["Ne"], N = 15788, Z=10, beta= 1/2, xc="LDA", r_min=1e-6, r_max=35, tol=1e-6)
    Na = dft_solver(atoms["Na"], N = 15788, Z=11, beta= 1/2, xc="LDA", r_min=1e-6, r_max=30, tol=1e-6)
    Mg = dft_solver(atoms["Mg"], N = 15788, Z=12, beta= 1/2, xc="LDA", r_min=1e-6, r_max=30, tol=1e-6)

    He_ = dft_solver(atoms["H"], N = 15788, Z=2, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    Li_ = dft_solver(atoms["He"], N = 15788, Z=3, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    Be_ = dft_solver(atoms["Li"], N = 15788, Z=4, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    B_ = dft_solver(atoms["Be"], N = 15788, Z=5, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    C_ = dft_solver(atoms["B"], N = 15788, Z=6, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    N_ = dft_solver(atoms["C"], N = 15788, Z=7, beta= 1/2, xc="LDA", r_min=1e-6, r_max=50, tol=1e-6)
    O_ = dft_solver(atoms["N"], N = 15788, Z=8, beta= 1/2, xc="LDA", r_min=1e-6, r_max=45, tol=1e-6)
    F_ = dft_solver(atoms["O"], N = 15788, Z=9, beta= 1/2, xc="LDA", r_min=1e-6, r_max=40, tol=1e-6)
    Ne_ = dft_solver(atoms["F"], N = 15788, Z=10, beta= 1/2, xc="LDA", r_min=1e-6, r_max=35, tol=1e-6)
    Na_ = dft_solver(atoms["Ne"], N = 15788, Z=11, beta= 1/2, xc="LDA", r_min=1e-6, r_max=30, tol=1e-6)
    Mg_ = dft_solver(atoms["Na"], N = 15788, Z=11, beta= 1/2, xc="LDA", r_min=1e-6, r_max=30, tol=1e-6)


    atoms = [H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg]
    cations = [He_, Li_, Be_, B_, C_, N_, O_, F_, Ne_, Na_, Mg_]

    for atom in atoms:
        print(name(atom))
        atom.solve(25)
    
    for cation in cations:
        print(name(cation).replace("_", "+"))
        cation.solve(25)

