import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
from numpy.linalg import eigvals
import pandas as pd

params = {
    'lambda1': 0.9,
    'lambda2': 0.8,
    'mu1': 0.3,
    'mu2': 0.3,
    'K': 1e6,
    'k1': 1e9,
    'k2': 1e9,
    'beta1': 470*60*24,
    'beta2': 70*60*24,
    'beta3': 240*60*24,
    'alpha1': 940*60*24,
    'alpha2': 510*60*24,
    'gamma': 2 }

# =================================================
# phase protrait of the mF-M cell dynamic system
# =================================================
# full_ODE
def full_ode(y):
    mF, M, CSF, PDGF = y
    p = params

    dmF = mF*( (p['lambda1']*PDGF/(p['k1'] + PDGF))*(1 - mF/p['K']) - p['mu1'] )
    dM = M*( p['lambda2']*CSF / (p['k2'] + CSF) - p['mu2'] )
    dcsf = p['beta1']*mF - p['alpha1']*M*(CSF/(p['k2']+CSF)) - p['gamma']*CSF
    dpdgf = p['beta2']*M + p['beta3']*mF - p['alpha2']*mF*(PDGF/(p['k1']+PDGF)) - p['gamma']*PDGF

    return [dmF, dM, dcsf, dpdgf]


# steady-state approximations (CSF & PDGF)
def csf_steady(mF, M, p=params):
    # restruct the equation: -gamma*CSF^2 + (beta1*mF-alpha1*M-gamma*k2)*CSF + beta1*mF*k2 = 0
    # a*CSF^2 + b*CSF + c = 0
    a = -p['gamma']
    b = p['beta1']*mF - p['alpha1']*M - p['gamma']*p['k2']
    c = p['beta1']*mF*p['k2']

    roots = np.roots([a, b, c])
    real_roots = [r.real for r in roots if np.isreal(r) and r.real >= 0]
    return real_roots if real_roots else []

def pdgf_steady(mF, M, p=params):
    # restruct the equation: -gamma*PDGF^2 + (beta2*M+beta3*mF-alpha2*mF-gamma*k1)*PDGF + k1*(beta2*M+beta3*mF) = 0
    # a*PDGF^2 + b*PDGF + c = 0
    a = -p['gamma']
    b = p['beta2']*M + p['beta3']*mF - p['alpha2']*mF - p['gamma']*p['k1']
    c = p['k1']*(p['beta2']*M + p['beta3']*mF)

    roots = np.roots([a, b, c])
    real_roots = [r.real for r in roots if np.isreal(r) and r.real >= 0]
    return real_roots if real_roots else []

# mF_M ode system
def mf_m_ode_system(mF, M, p=params):
    csf = csf_steady(mF, M, p)
    pdgf = pdgf_steady(mF, M, p)
    if csf and pdgf:
        dmF = mF * ((p['lambda1'] * pdgf[0] / (p['k1'] + pdgf[0])) * (1 - mF / p['K']) - p['mu1'])
        dM = M * (p['lambda2'] * csf[0] / (p['k2'] + csf[0]) - p['mu2'])
        return [dmF, dM]
    else:
        return [0, 0]
    
# create log-scale mesh grid
x_log = np.linspace(0, 7, 100)
y_log = np.linspace(0, 7, 100)
X_log, Y_log = np.meshgrid(x_log, y_log)
# convert log10 values back to linear scale
X = 10 ** X_log
Y = 10 ** Y_log
# initialize empty arrays
U = np.zeros_like(X)
V = np.zeros_like(Y)

# compute the vector field at each point on the mF-M grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u, v = mf_m_ode_system(X[i, j], Y[i, j], p = params)
        U[i, j] = u
        V[i, j] = v

# normalization
N = np.sqrt(U**2 + V**2)
U /= (N + 1e-7)
V /= (N + 1e-7)

# plot streamplot
plt.figure(figsize=(8, 6))
plt.streamplot(X_log, Y_log, U, V, color='gray', linewidth=1, density=1.2)
plt.xlabel('log10(mF)')
plt.ylabel('log10(M)')
plt.title('Streamplot of mF-M Dynamic System (in log space)')
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# find all fixed points：to find intersections of nullclines
# ==========================================================
# mF-nullcline: dmF/dt = 0
def nullclines_mF(mF, p=params):
    pdgf = (p['mu1']*p['k1']*p['K'])/(p['lambda1']*p['K'] - p['mu1']*p['K'] - p['lambda1']*mF)
    if pdgf <= 0:
        return None
    M = (1/p['beta2'])*(p['alpha2']*mF*pdgf/(p['k1']+pdgf) + p['gamma']*pdgf - p['beta3']*mF)
    return M if M >= 0 else None

# M-nullcline: dM/dt = 0
def nullclines_M(mF, p=params):
    csf = p['k2']*p['mu2'] / (p['lambda2']-p['mu2'])
    if csf <= 0:
        return None
    M = ((p['k2']+csf) / (p['alpha1']*csf)) * (p['beta1']*mF-p['gamma']*csf)
    return M if M >=0 else None

def compute_fixed_points(p=params):

    def nullclines_diff(mF, p):
        m1 = nullclines_mF(mF, p)
        m2 = nullclines_M(mF, p)
        if m1 is not None and m2 is not None:
            return m2 - m1
        else:
            return np.nan

    mF_space = np.logspace(0, 7, 50)
    diff_vals = np.array([nullclines_diff(mF, p) for mF in mF_space])

    # guesses where sign changes
    guess_list = []
    for i in range(len(diff_vals) - 1):
        if diff_vals[i] * diff_vals[i + 1] < 0:
            guess_list.append(mF_space[i])

    # compute fixed points
    fixed_points = []
    for guess_point in guess_list:
        try:
            mF_sol = fsolve(nullclines_diff, guess_point, args=(p))[0]
            M_sol = nullclines_mF(mF_sol, p)
            if mF_sol > 0 and M_sol > 0:
                fixed_points.append((mF_sol, M_sol))
        except:
            continue

    return fixed_points

fixed_points = compute_fixed_points()

# Plot nullclines and fixed points
mF_vals = np.logspace(0, 7, 500)
mF_null = [nullclines_mF(mf, params) for mf in mF_vals]
M_null = [nullclines_M(mf, params) for mf in mF_vals]

plt.figure(figsize=(8, 6))
plt.plot(mF_vals, mF_null, label='mF-nullcline (dmF/dt=0)')
plt.plot(mF_vals, M_null, label='M-nullcline (dM/dt=0)')
fp_arr = np.array(compute_fixed_points())
if len(fp_arr) > 0:
    plt.scatter(fp_arr[:, 0], fp_arr[:, 1], color='red', label='Fixed Points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mF')
plt.ylabel('M')
plt.legend()
plt.title('Nullclines and Fixed Points')
plt.grid(True)
plt.tight_layout()
plt.show()

# fail to identify the cold fibrosis fixed point:
# nullclines methods only solves for positive intersections, does not scan near M = 0 case
# create a new function of cold fibrosis
def cold_fibrosis_point(p=params):
    # dmF/dt = 0 when M = 0, dpdgf/dt = 0 when M = 0
    # combine two equation together: solve the cubic equation
    a = -p['gamma']
    b = ((p['K']/p['lambda1']) * (p['lambda1'] - p['mu1']) * (p['beta3'] - p['alpha2'])) - p['gamma']*p['k1']
    c = (p['K'] * p['k1'] / p['lambda1']) * (p['beta3']*p['lambda1'] - 2*p['mu1']*p['beta3'] + p['mu1']*p['alpha2'])
    d = -p['k1']**2 * p['mu1'] * p['K'] * p['beta3'] / p['lambda1']

    pdgf_roots = np.roots([a, b, c, d])
    
    cf_fixed_points = []
    for pdgf in pdgf_roots:
        if np.isreal(pdgf) and pdgf >= 0:
            pdgf = pdgf.real
            mF = p['K'] * ((p['lambda1'] - p['mu1']) / p['lambda1'] - (p['mu1'] * p['k1']) / (p['lambda1'] * pdgf))
            if mF >= 0:
                cf_fixed_points.append((mF, 0))

    return cf_fixed_points
cf_fixed_points = cold_fibrosis_point()
print("Cold fibrosis fixed points:", cf_fixed_points)


# ==================================
# compute and plot the separatrix
# ==================================
print("Number of fixed points found:", len(fixed_points))

# sort the fixed points in ascending order of their mF values
fixed_points_sorted = sorted(fixed_points, key=lambda x: x[0])
unstable_fixed_point = fixed_points_sorted[0]
hot_fibrosis_fixed_point = fixed_points_sorted[1]

# computes the Jacobian matrix of a 2D vector field: to check the type of fixed point
def jacobian(function, point, eps=1e-6):
    mF, M = point
    
    f0 = np.array(function([mF, M]))

    f1 = np.array(function([mF + eps, M]))
    df_dmF = (f1 - f0) / eps

    f2 = np.array(function([mF, M + eps]))
    df_dM = (f2 - f0) / eps
    J = np.column_stack([df_dmF, df_dM])
    return J

J = jacobian(mf_m_ode_system, unstable_fixed_point)
eigvals(J) # different sign: saddle point!


def mf_m_ode(state, t=0, p=params):
    mF, M = state
    csf_roots = csf_steady(mF, M, p)
    pdgf_roots = pdgf_steady(mF, M, p)

    if csf_roots and pdgf_roots:
        CSF = csf_roots[0]
        PDGF = pdgf_roots[0]
        dmF = mF * ((p['lambda1'] * PDGF / (p['k1'] + PDGF)) * (1 - mF / p['K']) - p['mu1'])
        dM = M * (p['lambda2'] * CSF / (p['k2'] + CSF) - p['mu2'])
        return [dmF, dM]
    else:
        return [0, 0]

# compute reversed-time ODE for separatrix trajectories by backward integration
def reversed_ode(state, t, p=params):
    dmF, dM = mf_m_ode(state, t, p)
    return [-dmF, -dM]

# integrate backward in time from small perturbations of the saddle point to compute separatrix trajectories
t_span = np.linspace(0, 100, 1000)
eps = 1e-3
separatrix_left = odeint(reversed_ode, [unstable_fixed_point[0] - eps, unstable_fixed_point[1] + eps], t_span)
separatrix_right = odeint(reversed_ode, [unstable_fixed_point[0] + eps, unstable_fixed_point[1] - eps], t_span)

# plot all together
plt.figure(figsize=(8, 6))
plt.streamplot(X_log, Y_log, U, V, color='gray', linewidth=1, density=1.2)
plt.plot(np.log10(separatrix_left[:, 0]), np.log10(separatrix_left[:, 1]), 'r--', label='Separatrix Left')
plt.plot(np.log10(separatrix_right[:, 0]), np.log10(separatrix_right[:, 1]), 'r--', label='Separatrix Right')
for fp in fixed_points:
    plt.plot(np.log10(fp[0]), np.log10(fp[1]), 'ko')
plt.plot(0, 0, 'bo', label='Healing Point')
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xlabel('log10(mF)')
plt.ylabel('log10(M)')
plt.title('Streamplot with Separatrix')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()