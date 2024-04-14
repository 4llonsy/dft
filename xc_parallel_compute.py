from sympy import symbols, pi, Rational, cbrt, sqrt, Function, ln, atan, diff, N
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

p_values = {'A': 0.0310907, 'x0': -0.10498, 'b': 3.72744, 'c': 12.9352}
f_values = {'A': 0.01554535, 'x0': -0.32500, 'b': 7.06042, 'c': 18.0578}
a_values = {'A': -1/(6*pi**2), 'x0': -0.00475840, 'b': 1.13107, 'c': 13.0045}
fpp_0 = 1.709921

nn, A, b, c, x0, i = symbols('n A b c x0 i')
r_s = cbrt(3/(4*pi*nn))
ex_p = -3*cbrt(9/(32*pi**2))/r_s/2
ex_f = -3*cbrt(18/(32*pi**2))/r_s/2
vx_p = diff(nn*ex_p, nn)
vx_f = diff(nn*ex_f, nn)

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
    return expression.subs(nn, value).n(digits)

def process_chunk(arr, expression, digits):
    return [sympy_function(j, expression, digits) for j in arr]

def setup_parallel_computation(expression):
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

    def f(self, z):
        return 

    def find_ex_Vx(self):
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
    
    def __init__(self, n_up, n_down, tol, xc):
        self.n = n_up + n_down
        z = (n_up - n_down)/(n_up + n_down)
        z = np.nan_to_num(z, 0)
        self.z = z
        self.z4 = z**4
        self.fz = ((1+z)**(4/3)+(1-z)**(4/3)-2)/(2*2**(1/3)-2)
        self.tol = tol
        self.digits = int(-np.log(tol))
        self.xc = xc