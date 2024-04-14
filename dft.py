import numpy as np
from scipy.integrate import odeint, simpson, trapezoid, solve_ivp
from scipy.optimize import toms748, ridder, bisect, brentq
from scipy.interpolate import interp1d
from xc_parallel_compute import xc_calculator

def parabola_2a(x1, x2, x3, y1, y2, y3):
    return 2*(y1/(x1-x3)/(x1-x2) + y2/(x2-x3)/(x2-x1) + y3/(x3-x1)/(x3-x2))

def double_derivative(y,x):
    n = len(x)
    arr = [parabola_2a(x[i], x[i+1], x[i+2], y[i], y[i+1], y[i+2]) for i in range(n-2)]
    return np.array([arr[0]]+arr+[arr[-1]])

class orbital_eigenvalue_solver():

    def diff_eq(self, y, r, E, l):
        y0, y1 = y
        dydx = [y1, -2*(self.Z*y0/r -self.Vh_interp(r)*y0 -self.Vx_interp(r)*y0 -self.Vc_interp(r)*y0 + E*y0 - l*(l+1)*y0/r**2/2)]
        return dydx

    def differential_solver(self, E, l):
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
        find_eigenroot = lambda E: self.differential_solver(E, l)[-1]
        n = 4*self.Z
        alpha = (E_min-E_max)/n**2
        arr = [alpha*i**2+E_max for i in np.arange(n, 0, -1)]
        for i in range(len(arr)-1):
            try:
                a = find_eigenroot(arr[i])
                b = find_eigenroot(arr[i+1])
                if np.sign(a)!=np.sign(b):
                    E = toms748(find_eigenroot, arr[i], arr[i+1], xtol=precision/10) 
                    Y = self.differential_solver(E, l)
                    return Y[0][::-1], E
            except ValueError:
                continue
        return None

    def __init__(self, Z, r, Vh, Vx, Vc, qmax):
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

class poisson_solver():

    def diff_eq(self, y, x):
        y0, y1 = y
        dydx = [y1, -4*np.pi*x*self.n_interp(x)]
        return dydx

    def find_Vh(self):
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

    def __init__(self, r, n_up, n_down, qmax):
        self.n = n_up + n_down
        self.r = r
        self.qmax = qmax
        self.n_interp = interp1d(self.r, self.n, kind='linear', fill_value="extrapolate")


class dft_solver():

    def __init__(self, parameters, N, Z, beta, xc="lda", r_min=1e-5, r_max=50, tol=1e-6):
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
                    u, epsilon = a.find_eigenvalue(l, self.E_min-np.sqrt(self.tol), self.E_max, self.tol)
                    #self.E_min = (3*self.E_min+epsilon)/4-2*self.tol
                else:
                    emin = eigenvalues[f"{n} {l-1}"]
                    u, epsilon = a.find_eigenvalue(l, emin-np.sqrt(self.tol), self.E_max, self.tol)
            else:
                emin = eigenvalues[f"{n-1} {l}"]
                u, epsilon = a.find_eigenvalue(l, emin+np.sqrt(self.tol), self.E_max, self.tol)
            T += -occupancy*simpson((double_derivative(u,self.r)*u), self.r)/2
            if l!=0:
                T += occupancy*l*(l+1)*simpson(u**2/self.r**2, self.r)/2
            eigenvalues[f"{n} {l}"] = epsilon
            n_up += up*u**2/self.r**2/4/np.pi
            E_e += occupancy*epsilon
            n_down += down*u**2/self.r**2/4/np.pi

        return eigenvalues, n_up, n_down, T, E_e

    def solve(self, iterations):
        if self.xc.lower()=="lda" or self.diamagnetic:
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
            return None
        return None
