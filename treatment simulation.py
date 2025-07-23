from my_function import *
from signal_simulation import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path

# load the previously saved separatrix data
data = pd.read_csv('separatrix.csv')
separatrix = data[['mF', 'M']].values
separatrix = separatrix[np.argsort(separatrix[:, 0])]

# function to determine if a point lies in the healing basin
def below_separatrix(mF, M, separatrix):
    if mF <= 0 or M <= 0:
        return True
    
    sep_log_mF = np.log10(separatrix[:, 0])
    sep_log_M = np.log10(separatrix[:, 1])
    
    log_mF = np.log10(mF)

    # if the point outside the healing basin: fibrosis
    if log_mF > sep_log_mF[-1]:
        return False
    
    # interpolate separatrix M at the current mF value
    log_M_interp = np.interp(log_mF, sep_log_mF, sep_log_M)
    return np.log10(M) < log_M_interp

# =======================================
# single parameter sensitivity analysis
# =======================================
folds = [0.5, 1, 2]  # define fold change to scan
param_names = ['alpha1', 'alpha2', 'beta1', 'beta3', 'lambda1', 'mu1']
t = np.linspace(0, 80, 500)
y0 = [1, 1] # initial condition

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# loop over each parameter
for idx, param in enumerate(param_names):
    trajectories = []
    end_points = []
    status = []

    # loop over each fold change
    for i, fold in enumerate(folds):
        new_params = params.copy()
        new_params[param] *= fold
        # simulate the system with prolonged type
        sol = odeint(modified_mf_m_ode, y0, t, args=(new_params, 'prolonged'))
        trajectories.append(sol)
        end_points.append(sol[-1])
        status.append(below_separatrix(sol[-1, 0], sol[-1, 1], separatrix))

    # plot trajectories
    ax = axes[idx]
    for i, sol in enumerate(trajectories):
        label = f"{folds[i]:.2f}×{param}"
        # set green healed and red fibrosis
        color = 'green' if status[i] else 'red'
        ax.plot(np.log10(sol[:, 0]), np.log10(sol[:, 1]), label=label, color=color)
    # add separatrix
    ax.plot(np.log10(separatrix[:, 0]), np.log10(separatrix[:, 1]), 'k--', label='Separatrix')
    ax.set_title(f'Trajectories with {param}')
    ax.set_xlabel('log10(mF)')
    ax.set_ylabel('log10(M)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.legend()

plt.tight_layout(pad=2.5)
plt.show()

# =====================================================
# 2D parameter sweep of alpha2, beta3, lambda1 and mu1
# =====================================================
# define all possible parameter pairs to scan
param_pairs = [
    ('alpha2', 'beta3'),
    ('alpha2', 'lambda1'),
    ('alpha2', 'mu1'),
    ('beta3', 'lambda1'),
    ('beta3', 'mu1'),
    ('lambda1', 'mu1')
]

# define fold change to scan
folds = np.linspace(0.5, 2.0, 10)  
t = np.linspace(0, 80, 500)
y0 = [1, 1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# store all maps for each parameter pair
fate_maps = []

# loop over each parameter pair
for idx, (param_x, param_y) in enumerate(param_pairs):
    fate_map = np.zeros((len(folds), len(folds)), dtype=bool)
    
    # scan both parameters over fold changes
    for i, x in enumerate(folds):
        for j, y in enumerate(folds):
            # create new parameter dictionary
            new_p = params.copy()
            new_p[param_x] *= x
            new_p[param_y] *= y

            # simulate ODE with prolonged simulation
            sol = odeint(modified_mf_m_ode, y0, t, args=(new_p, 'prolonged'))
            # find final state
            mF_final, M_final = sol[-1]
            # check whether the final state is within healing basin
            fate_map[i, j] = below_separatrix(mF_final, M_final, separatrix)

    fate_maps.append(fate_map)

    X, Y = np.meshgrid(folds, folds)
    ax = axes[idx]
    # green: healing = 1, white: fibrosis = 0
    cmap = plt.get_cmap('Greens')

    im = ax.pcolormesh(X, Y, fate_map, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax.set_xlabel(f'{param_x} fold')
    ax.set_ylabel(f'{param_y} fold')
    ax.set_title(f'{param_x} vs {param_y}')
    
plt.colorbar(im, label='Healing (1=True, 0=False)')
plt.tight_layout(pad=2.5)
plt.show()


# =======================================
# compute area under the separatrix
# =======================================
def compute_healing_area(separatrix, grid_x, grid_y):
    # create polygon path from the separatrix boundary
    boundary_path = Path(separatrix)
    # flatten the grid into 2D 
    points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    # check if the grid lie inside the separatrix region
    inside = boundary_path.contains_points(points)
    inside_mask = inside.reshape(grid_x.shape)

    # compute the area of each grid
    dx = np.abs(grid_x[0,1] - grid_x[0,0])
    dy = np.abs(grid_y[1,0] - grid_y[0,0])
    # compute unit area
    unit_area = dx * dy
    # total area: number of points inside * unit area
    healing_area = np.sum(inside_mask) * unit_area

    return healing_area

x_vals = np.linspace(-3, 8, 200)
y_vals = np.linspace(-3, 8, 200)
grid_x, grid_y = np.meshgrid(x_vals, y_vals)

sep_log_mF = np.log10(separatrix[:, 0])
sep_log_M = np.log10(separatrix[:, 1])
sep_log = np.column_stack((sep_log_mF, sep_log_M)) # shape (N, 2)

# compute the original healing area
original_area = compute_healing_area(sep_log, grid_x, grid_y)
print(original_area)

# ========================
# alpha_2 vs beta_3
# ========================
area_matrix = np.full((len(folds), len(folds)), np.nan) 
t_span = np.linspace(0, 200, 2000)
eps = 1e-3

for i, a2_fold in enumerate(folds):
    for j, b3_fold in enumerate(folds):
        params_new = params.copy()
        params_new['alpha2'] *= a2_fold
        params_new['beta3'] *= b3_fold
        
        fixed_points = compute_fixed_points(params_new)
        cf_points = cold_fibrosis_point(params_new)
        all_fixed_points = combine_fixed_points(fixed_points, cf_points)

        if not all_fixed_points:
            continue

        def find_saddle_points(fixed_points, p):
            saddles = []
            for fp in fixed_points:
                J = jacobian(lambda state: mf_m_ode_wrapper(state, p), fp, eps=1e-6)
                eigvals = np.linalg.eigvals(J)
                if np.any(eigvals.real > 0) and np.any(eigvals.real < 0):
                    saddles.append(fp)
            return saddles

        unstable_fixed_points = find_saddle_points(all_fixed_points, params_new)
        if unstable_fixed_points is None:
            continue

        all_sep = []
        for saddle in unstable_fixed_points:
            left = odeint(reversed_ode, [saddle[0] - eps, saddle[1] + eps], t_span, args=(params_new,))
            right = odeint(reversed_ode, [saddle[0] + eps, saddle[1] - eps], t_span, args=(params_new,))
            sep = np.vstack([left[::-1], right])
            sep = sep[np.all(sep > 0, axis=1)]
            all_sep.append(sep)
        
        sep_all = np.vstack(all_sep)
        sep_log = np.log10(sep_all)

        area = compute_healing_area(sep_log, grid_x, grid_y)
        area_matrix[i, j] = area

ratio_matrix = area_matrix / original_area

plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(folds, folds)
figure = plt.pcolormesh(X, Y, ratio_matrix, cmap='coolwarm')
plt.colorbar(figure, label='Healing Area Ratio (new / original)')
plt.xlabel('alpha2 fold')
plt.ylabel('beta3 fold')
plt.title('Healing Basin Area Ratio Heatmap')
plt.tight_layout()
plt.show()

# ============ test ==============
original_sep = pd.read_csv('separatrix.csv')[['mF', 'M']].values
params_base = params.copy()
params1 = params_base.copy()
params1['alpha2'] *= 0.5
params1['beta3'] *= 1.0
generate_separatrix_plot(params1, original_sep)
print(len(fixed_points))