from sympy import symbols, pi, Rational, cbrt, sqrt, Function, ln, atan, diff, N
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

"""
Credits to ChatGPT for generating the documentation in the code
"""

# Define constants and parameters for the exchange-correlation functional
p_values = {'A': 0.0310907, 'x0': -0.10498, 'b': 3.72744, 'c': 12.9352}
f_values = {'A': 0.01554535, 'x0': -0.32500, 'b': 7.06042, 'c': 18.0578}
a_values = {'A': -1/(6*pi**2), 'x0': -0.00475840, 'b': 1.13107, 'c': 13.0045}
fpp_0 = 1.709921 #f''(0) used in calculation

# Define symbols for symbolic computations
nn, A, b, c, x0, i = symbols('n A b c x0 i')

# Calculate electron density related parameter rs (electron gas parameter)
r_s = cbrt(3/(4*pi*nn))

# Defined the equations for exchange potentials
ex_p = -3*cbrt(9/(32*pi**2))/r_s/2
ex_f = -3*cbrt(18/(32*pi**2))/r_s/2
vx_p = diff(nn*ex_p, nn)
vx_f = diff(nn*ex_f, nn)

# Defined the equations for correlation potentials
x = sqrt(r_s)
X = Function('X')(i)
X = i**2 + b*i + c
Q = sqrt(4*c - b**2)
F = A * (ln(x**2/X.subs(i,x)) + 2*b/Q*atan(Q/(2*x+b)) - (b*x0/X.subs(i,x0)) * (ln((x-x0)**2/X.subs(i,x)) + 2*(b+2*x0)/Q * atan(Q/(2*x+b))))
ec_p = F.subs(p_values)
ec_f = F.subs(f_values)
ec_a = F.subs(a_values)
vc_p = diff(nn*ec_p)
vc_f = diff(nn*ec_f)
vc_a = diff(nn*ec_a)


def sympy_function(value, expression, digits):
    """
    Evaluates a symbolic expression at a given value with specified precision.

    Parameters:
        value (float): The value at which to evaluate the expression.
        expression (sympy expression): The symbolic expression to evaluate.
        digits (int): The number of significant digits for the result.

    Returns:
        float: The evaluated expression at the specified value.
    """
    return expression.subs(nn, value).n(digits)

def process_chunk(arr, expression, digits):
    """
    Processes a chunk of values, evaluating a symbolic expression for each value.

    Parameters:
        arr (list): List of values at which to evaluate the expression.
        expression (sympy expression): The symbolic expression to evaluate.
        digits (int): The number of significant digits for each evaluation.

    Returns:
        list: The results of the evaluations.
    """
    return [sympy_function(j, expression, digits) for j in arr]

def setup_parallel_computation(expression):
    """
    Sets up a parallel computation environment for processing large arrays of values.

    Parameters:
        expression (sympy expression): The symbolic expression to evaluate.

    Returns:
        function: A function that takes an array of values and a precision, and returns the parallel computed results.
    """
    def parallel_computation(n, digits):
        num_cores = multiprocessing.cpu_count()
        num_processes = max(1, num_cores-1)
        chunk_size = int(np.ceil(len(n) / num_processes))
        
        split_arrays = [n[i:i + chunk_size] for i in range(0, len(n), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_results = [executor.submit(process_chunk, chunk, expression, digits) for chunk in split_arrays]
            
            for future in future_results:
                results.extend(future.result())
        
        return np.array(results)
    return parallel_computation

# Setup parallel computation functions for various functions
parallel_compute_vcp = setup_parallel_computation(vc_p)
parallel_compute_vcf = setup_parallel_computation(vc_f)
parallel_compute_vca = setup_parallel_computation(vc_a)

parallel_compute_ecp = setup_parallel_computation(ec_p)
parallel_compute_ecf = setup_parallel_computation(ec_f)
parallel_compute_eca = setup_parallel_computation(ec_a)

parallel_compute_exp = setup_parallel_computation(ex_p)
parallel_compute_exf = setup_parallel_computation(ex_f)

parallel_compute_vxp = setup_parallel_computation(vx_p)
parallel_compute_vxf = setup_parallel_computation(vx_f)

class xc_calculator():
    """
    A class designed to calculate exchange and correlation energies and potentials
    using parallel computation for Local Density Approximation (LDA) and 
    Local Spin Density (LSD) Approximations.

    Attributes:
        n (np.array): Total electron density, sum of spin-up and spin-down densities.
        z (np.array): Relative spin polarization, calculated as (n_up - n_down) / (n_up + n_down).
        z4 (np.array): Fourth power of relative spin polarization, used in LSD calculations.
        fz (np.array): Spin scaling factor, used in interpolation between spin-polarized and unpolarized results.
        tol (float): Tolerance for numerical calculations, affects precision of results.
        digits (int): Number of significant digits derived from tolerance for parallel computations.
        xc (str): Specifies the type of exchange-correlation functional used ('lda' or 'lsd').
    """
    def __init__(self, n_up, n_down, tol, xc):
        """
        Initializes the xc_calculator with electron densities, tolerance, and xc functional type.

        Parameters:
            n_up (np.array): Spin-up electron density.
            n_down (np.array): Spin-down electron density.
            tol (float): Numerical tolerance for computation.
            xc (str): Type of exchange-correlation functional ('lda' or 'lsd').
        """
        self.n = n_up + n_down
        z = (n_up - n_down)/(n_up + n_down)
        z = np.nan_to_num(z, 0)
        self.z = z
        self.z4 = z**4
        self.fz = ((1+z)**(4/3)+(1-z)**(4/3)-2)/(2*2**(1/3)-2)
        self.tol = tol
        self.digits = int(-np.log(tol))
        self.xc = xc

    def find_ex_Vx(self):
        """
        Computes exchange energy and potential for the given density and xc functional.

        Returns:
            tuple: (exchange energy, exchange potential) arrays calculated based on the xc functional.
        """
        n = self.n
        Vxp = np.nan_to_num(parallel_compute_vxp(n, self.digits))
        exp = np.nan_to_num(parallel_compute_exp(n, self.digits))
        if np.all(self.z)==0 or self.xc.lower() == "lda":
            return exp, Vxp
        if self.xc.lower()=="lsd":
            Vxf = np.nan_to_num(parallel_compute_vxf(n, self.digits))
            exf = np.nan_to_num(parallel_compute_exf(n, self.digits))
            Vx = Vxp + (Vxf-Vxp)*self.fz
            ex = exp + (exf-exp)*self.fz
            return ex, Vx
        return None

    def find_ec_Vc(self):
        """
        Computes correlation energy and potential for the given density and xc functional.

        Returns:
            tuple: (correlation energy, correlation potential) arrays calculated based on the xc functional.
        """
        n = self.n
        Vcp = np.nan_to_num(parallel_compute_vcp(n, self.digits))
        ecp = np.nan_to_num(parallel_compute_ecp(n, self.digits))
        if np.all(self.z)==0 or self.xc.lower() == "lda":
            return ecp, Vcp
        if self.xc.lower()=="lsd":
            Vcf = np.nan_to_num(parallel_compute_vcf(n, self.digits))
            ecf = np.nan_to_num(parallel_compute_ecf(n, self.digits))
            Vca = np.nan_to_num(parallel_compute_vca(n, self.digits))
            eca = np.nan_to_num(parallel_compute_eca(n, self.digits))
            Vc = Vcp*(1-self.fz*self.z4) + self.fz*(1-self.z4)*Vca/fpp_0 + self.fz*self.z4*Vcf
            ec = ecp*(1-self.fz*self.z4) + self.fz*(1-self.z4)*eca/fpp_0 + self.fz*self.z4*ecf
            return ec, Vc
        return None
    
