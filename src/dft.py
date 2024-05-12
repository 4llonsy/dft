import numpy as np
from scipy.integrate import odeint, simpson, solve_ivp
from scipy.optimize import toms748
from scipy.interpolate import interp1d
from xc_parallel_compute import xc_calculator

"""
Credits to ChatGPT for generating the documentation in the code
"""

def parabola_2a(x1, x2, x3, y1, y2, y3):
    """
    Computes twice the coefficient 'a' of a parabolic curve passing through three points.
    
    The parabolic equation can be represented as y = ax^2 + bx + c. This function calculates
    2*a using three points: (x1, y1), (x2, y2), (x3, y3), which satisfy the equation.

    Parameters:
        x1, x2, x3 (float): x-coordinates of the three points.
        y1, y2, y3 (float): y-coordinates of the three points.

    Returns:
        float: The value representing twice the coefficient 'a' of the parabola.
    """
    return 2*(y1/(x1-x3)/(x1-x2) + y2/(x2-x3)/(x2-x1) + y3/(x3-x1)/(x3-x2))

def double_derivative(y,x): 
    """
    Estimates the second derivative of a function at multiple points using parabolic interpolation.
    
    This function calculates the second derivative by fitting a parabola through every three
    consecutive points (x[i], y[i]), (x[i+1], y[i+1]), (x[i+2], y[i+2]) and then extracting
    the curvature (second derivative) from this parabola.

    Parameters:
        y (array-like): y-values of the function at corresponding x-values.
        x (array-like): x-values at which the function is evaluated.

    Returns:
        np.array: An array of the second derivatives at all x-values except the two endpoints.
                  The second derivative at the endpoints is approximated as the same as the
                  nearest calculated second derivative.
    """
    n = len(x)
    arr = [parabola_2a(x[i], x[i+1], x[i+2], y[i], y[i+1], y[i+2]) for i in range(n-2)]
    return np.array([arr[0]]+arr+[arr[-1]])


class orbital_eigenvalue_solver():
    """
    A solver for finding eigenvalues and eigenfunctions for the orbital equation in quantum mechanics
    considering different potentials (Vh, Vx, Vc) and angular momentum quantum number `l`.
    
    Attributes:
        Z (float): The atomic number or effective nuclear charge.
        r (array): Radial coordinate array, ordered in decreasing values for reverse integration.
        qmax (float): Maximum number of electrons.
        Vh (array): Hartree potential values corresponding to `r`.
        Vx (array): Exchange potential values corresponding to `r`.
        Vc (array): Correlation potential values corresponding to `r`.
    """

    def __init__(self, Z, r, Vh, Vx, Vc, qmax):
        """
        Initializes the solver with the atomic data and potential functions.

        Parameters:
            Z (float): The atomic number or effective nuclear charge.
            r (array): Radial distances.
            Vh (array): Hartree potential values.
            Vx (array): Exchange potential values.
            Vc (array): Correlation potential values.
            qmax (float): Number of electrons.
        """
        self.Z = Z
        self.r = r[::-1]
        self.qmax = qmax
        self.Vh = Vh[::-1]
        self.Vx = Vx[::-1]
        self.Vc = Vc[::-1]
        self.Vx_interp = interp1d(self.r, self.Vx, kind='linear', fill_value="extrapolate")
        self.Vh_interp = interp1d(self.r, self.Vh, kind='linear', fill_value="extrapolate")
        self.Vc_interp = interp1d(self.r, self.Vc, kind='linear', fill_value="extrapolate")
        self.data = dict()

    def diff_eq(self, y, r, E, l):
        """
        Differential equation for the radial part of the wavefunction in a potential field.

        Parameters:
            y (tuple): Tuple containing the current value of the wavefunction and its derivative (y, y').
            r (float): Current radial position.
            E (float): Energy eigenvalue.
            l (int): Angular momentum quantum number.

        Returns:
            list: First and second derivatives of the wavefunction at r.
        """
        y0, y1 = y
        dydx = [y1, -2*(self.Z*y0/r -self.Vh_interp(r)*y0 -self.Vx_interp(r)*y0 -self.Vc_interp(r)*y0 + E*y0 - l*(l+1)*y0/r**2/2)]
        return dydx

    def differential_solver(self, E, l):
        """
        Solves the differential equation for a given energy `E` and angular momentum `l`.

        Parameters:
            E (float): Energy eigenvalue.
            l (int): Angular momentum quantum number.

        Returns:
            tuple: Normalized wavefunction values, their derivatives, and value at origin.
        """
        k = np.sqrt(-2*E)
        y0 = np.exp(-k*self.r[-1])
        y1 = -k*y0
        # Create an array of x values from 10 to 0.1 (integration will go backward)
        y = odeint(self.diff_eq, [y0, y1], self.r, args=(E,l), atol=2e-12, rtol=2e-12)
        I = simpson(y[:, 0]**2, self.r)
        y = y / np.sqrt(-I)
        last0=y[:, 0][-1]
        last1=y[:, 1][-1]
        return y[:, 0], y[:, 1], last0-last1*self.r[-1] #Returns y(x) y'(x) and y(0)


    def find_eigenvalue(self, l, E_min, E_max, precision):
        """
        Finds an eigenvalue within a specified range by scanning and using root-finding.

        Parameters:
            l (int): Angular momentum quantum number.
            E_min (float): Minimum energy to consider.
            E_max (float): Maximum energy to consider.
            precision (float): Desired precision for the eigenvalue.

        Returns:
            tuple: Eigenfunction and corresponding eigenvalue if found, otherwise None.
        """
        find_eigenroot = lambda E: self.differential_solver(E, l)[-1]
        n = 4*self.Z
        alpha = (E_min-E_max)/n**2
        arr = [alpha*i**2+E_max for i in np.arange(n, 0, -1)]
        for i in range(len(arr)-1):
            try:
                a = find_eigenroot(arr[i])
                b = find_eigenroot(arr[i+1])
                if a*b<=0:
                    E = toms748(find_eigenroot, arr[i], arr[i+1], xtol=precision/10) 
                    Y = self.differential_solver(E, l)
                    return Y[0][::-1], E
            except ValueError:
                continue
        return None


class poisson_solver():
    """
    A solver for the Poisson equation specific to electrostatic potentials, typically used
    in the context of electronic structure calculations in physics and chemistry.
    
    Attributes:
        r (array): Array of radial grid
        n (array): Total electron density (sum of spin-up and spin-down densities).
        qmax (float): Boundary condition for the potential at large distances.
        n_interp (callable): An interpolator for electron density as a function of radius.
    """

    def __init__(self, r, n_up, n_down, qmax):
        """
        Initializes the Poisson solver with the required physical and numerical parameters.

        Parameters:
            r (array): Array of radial grid
            n_up (array): Electron density for spin-up electrons.
            n_down (array): Electron density for spin-down electrons.
            qmax (float): Value of the electrostatic potential at the farthest radial distance.
        """
        self.n = n_up + n_down
        self.r = r
        self.qmax = qmax
        self.n_interp = interp1d(self.r, self.n, kind='linear', fill_value="extrapolate")

    def diff_eq(self, y, x):
        """
        Defines the differential equation derived from Poisson's equation in spherical coordinates.

        Parameters:
            y (tuple): A tuple containing the current value of the potential (y[0]) and its first derivative (y[1]).
            x (float): The current radial position.

        Returns:
            list: Derivatives of the potential and its slope at the point x.
        """
        y0, y1 = y
        dydx = [y1, -4*np.pi*x*self.n_interp(x)]
        return dydx

    def find_Vh(self):
        """
        Solves the Poisson equation to find the Hartree potential, Vh, as a function of radius.

        Returns:
            array: Computed Hartree potential (Vh)
        """
        y0 = 0  # y(x0)
        y1 = 1
        r = self.r
        # Solve the differential equation
        y = odeint(self.diff_eq, [y0, y1], r)
        y = y[:, 0]
        #slope = (y[-2]-y[-1])/((r[-2]-r[-1]))
        alpha = (self.qmax - y[-1])/r[-1]
        y = y+alpha*r 
        return y/r


class dft_solver():
    """
    A solver class for Density Functional Theory (DFT) calculations, handling the computation
    of electronic structures.

    Attributes:
        r (np.array): Radial grid points for computation.
        E_min (float): Minimum energy boundary for eigenvalue calculations.
        E_max (float): Maximum energy boundary for eigenvalue calculations.
        N (int): Number of radial grid points.
        tol (float): Tolerance for numerical methods.
        Z (int): Atomic number, representing the nuclear charge.
        table (list): Quantum number configurations and occupations.
        qmax (float): Maximum electronic charge considered in the system.
        beta (float): Mixing parameter for potential updating in iterative solvers.
        xc (str): Exchange-correlation functional type ('lda' or 'lsd').
        diamagnetic (bool): Indicates whether all spins are paired.
    """

    def __init__(self, parameters, N, Z, beta, xc="lda-vwn", r_min=1e-5, r_max=50, tol=1e-6):
        """
        Initializes the DFT solver with the specified parameters and constructs the radial grid.

        Parameters:
            parameters (dict): Configuration dictionary containing quantum numbers and occupations.
            N (int): Number of grid points.
            Z (int): Atomic number.
            beta (float): Mixing parameter for iterative potential updates.
            xc (str): Exchange-correlation functional.
            r_min (float): Minimum radial distance.
            r_max (float): Maximum radial distance.
            tol (float): Tolerance for numerical calculations.
        """
        l={"s": 0, "p": 1, "d": 2, "f": 3}
        self.r = np.array([r_min*(r_max/r_min)**(i/(N-1)) for i in range(N)])
        self.E_min=-Z**2*0.5
        self.E_max=-tol
        self.N=N
        self.tol=tol
        self.Z=Z
        self.table = [(int(o[0]), l[o[1]], int(n[0]), int(n[2])) for o, n in parameters.items()]
        self.qmax = sum([i + j for _, _, i, j in self.table])
        self.beta = beta
        self.xc = xc
        self.diamagnetic = np.all(np.array([i == j for _, _, i, j in self.table]))

    def find_e_nup_ndown_T(self, Vh, Vx, Vc):
        """
        Calculates the electron densities (spin-up and spin-down), eigenvalues, and kinetic and total energies.

        Parameters:
            Vh (np.array): Hartree potential array.
            Vx (np.array): Exchange potential array.
            Vc (np.array): Correlation potential array.

        Returns:
            tuple: A tuple containing the calculated eigenvalues, spin-up and spin-down densities, 
                   total kinetic energy, and electronic energy.
        """
        n_up = np.zeros_like(self.r)
        n_down = np.zeros_like(self.r)
        eigenvalues = dict()
        T = 0
        E_e = 0
        for n, l, up, down in self.table:
            occupancy = up+down
            a = orbital_eigenvalue_solver(self.Z, self.r, Vh, Vx, Vc, self.qmax)
            if (n-l)==1:
                if (n==1 and l==0):
                    u, epsilon = a.find_eigenvalue(l, self.E_min-0.001, self.E_max, self.tol)
                    #self.E_min = (3*self.E_min+epsilon)/4-2*self.tol
                else:
                    emin = eigenvalues[f"{n} {l-1}"]
                    u, epsilon = a.find_eigenvalue(l, emin-0.001, self.E_max, self.tol)
            else:
                emin = eigenvalues[f"{n-1} {l}"]
                u, epsilon = a.find_eigenvalue(l, emin+0.001, self.E_max, self.tol)
            T += -occupancy*simpson((double_derivative(u,self.r)*u), self.r)/2
            if l!=0:
                T += occupancy*l*(l+1)*simpson(u**2/self.r**2, self.r)/2
            eigenvalues[f"{n} {l}"] = epsilon
            n_up += up*u**2/self.r**2/4/np.pi
            E_e += occupancy*epsilon
            n_down += down*u**2/self.r**2/4/np.pi

        return eigenvalues, n_up, n_down, T, E_e

    def solve(self, iterations):
        """
        Iteratively solves for the ground state energy using a self-consistent field approach. 
        Prints all the calculated energies every iteration.

        Parameters:
            iterations (int): Number of iterations to perform.

        Returns:
            float: Converged ground state energy
        """
        if self.xc.lower()=="lda-vwn" or self.diamagnetic:
            a, b, E_new, E_e, E_h, E_x, E_c, E_enuc, T, E = np.zeros(10)
            Vh = np.zeros_like(self.r)
            Vx = np.zeros_like(self.r)
            Vc = np.zeros_like(self.r)
            beta_arr=np.linspace(self.beta, 1, iterations)
            for i in range(iterations):
                beta = beta_arr[i]
                print(f"Iteration {i}")
                print("Energies:")
                print("E_tot: {:.6f},E_kin: {:.6f},E_coul: {:.6f},E_enuc: {:.6f},E_xc: {:.6f},e: {:.6f}".replace(",", "\n").format(E_new, T, E_h, E_enuc, E_x+E_c, E_e))
                E = E_new
                eigenvalues, n_up, n_down, T, E_e = self.find_e_nup_ndown_T(Vh, Vx, Vc)
                n = n_up+n_down
                print("Eigenvalues:")
                tablel={"0": "s", "1": "p", "2": "d", "3": "f"}
                for key, value in eigenvalues.items():
                    l = tablel[key[-1]]
                    print("{:1}{:1}: {:.6f}". format(key[0], l, value))
                b = poisson_solver(self.r, n_up, n_down, self.qmax)
                xc = xc_calculator(n_up, n_down, self.tol, self.xc)
                Vh = beta*b.find_Vh() + Vh*(1-beta)
                ex, Vxn = xc.find_ex_Vx()
                ec, Vcn = xc.find_ec_Vc()
                Vx = Vxn*beta+(1-beta)*Vx
                Vc = Vcn*beta+(1-beta)*Vc
                E_h = 4*np.pi*simpson(Vh*n*self.r**2, self.r)/2
                E_x = 4*np.pi*simpson(ex*n*self.r**2, self.r)
                E_c = 4*np.pi*simpson(ec*n*self.r**2, self.r)
                E_enuc = 4*np.pi*simpson(-self.Z*n*self.r, self.r)
                E_new = T+E_enuc+E_h+E_x+E_c
            print("Converged values:")
            print("E_tot: {:.6f},E_kin: {:.6f},E_coul: {:.6f},E_enuc: {:.6f},E_xc: {:.6f},e: {:.6f}".replace(",", "\n").format(E_new, T, E_h, E_enuc, E_x+E_c, E_e))
            print("\n")
            return  E_new

        if self.xc.lower()=="lsd":
            """LSD approximation not yet implemented"""
            return None
        return None
