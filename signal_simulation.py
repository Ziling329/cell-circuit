from original_version import *
import numpy as np

# build heaviside function
def heaviside_function(t, starts=[0], durations=[2], amplitude=1e6):
    return sum(amplitude * (t >= start) * (t < start + duration) for start, duration in zip(starts, durations))

# ode system with pulse input
def modified_mf_m_ode(y, t, p=params, pulse_type='transient'):
    mF, M = y
    if pulse_type == 'transient':
        I_t = heaviside_function(t, starts=[0], durations=[2], amplitude=1e6)
    elif pulse_type == 'repetitive':
        I_t = heaviside_function(t, starts=[0, 4], durations=[2, 2], amplitude=1e6)
    elif pulse_type == 'prolonged':
        I_t = heaviside_function(t, starts=[0], durations=[4], amplitude=1e6)
    else:
        I_t = 0

    csf = csf_steady(mF, M, p)
    pdgf = pdgf_steady(mF, M, p)
    if csf and pdgf:
        dmF = mF * ((p['lambda1'] * pdgf[0] / (p['k1'] + pdgf[0])) * (1 - mF / p['K']) - p['mu1'])
        dM = I_t + M * (p['lambda2'] * csf[0] / (p['k2'] + csf[0]) - p['mu2'])
        return [dmF, dM]
    else:
        return [0, 0]
    

# solve odes
y0 = [1, 1]
t = np.linspace(0, 80, 500)
sol_transient = odeint(modified_mf_m_ode, y0, t, args=(params, 'transient'))
sol_repetitive = odeint(modified_mf_m_ode, y0, t, args=(params, 'repetitive'))
sol_prolonged = odeint(modified_mf_m_ode, y0, t, args=(params, 'prolonged'))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # plot results
    plt.figure(figsize=(15, 6))

    # M
    plt.subplot(1, 2, 1)
    plt.plot(t, sol_transient[:, 1], label='M (Transient)', color='green')
    plt.plot(t, sol_repetitive[:, 1], label='M (Repetitive)', color='red')
    plt.plot(t, sol_prolonged[:, 1], label='M (Prolonged)', color='blue')
    plt.xlabel('Time (days)')
    plt.ylabel('cells per ml')
    plt.title('Cell Dynamics (macrophages)')
    plt.legend()
    plt.grid(True)

    # mF
    plt.subplot(1, 2, 2)
    plt.plot(t, sol_transient[:, 0], label='mF (Transient)', color='green')
    plt.plot(t, sol_repetitive[:, 0], label='mF (Repetitive)', color='red')
    plt.plot(t, sol_prolonged[:, 0], label='mF (Prolonged)', color='blue')
    plt.xlabel('Time (days)')
    plt.ylabel('cells per ml')
    plt.title('Cell Dynamics (myofibroblasts)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # plot trajectories
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(np.log10(separatrix_left[:, 0]), np.log10(separatrix_left[:, 1]), 'r--', label='Separatrix Left')
    plt.plot(np.log10(separatrix_right[:, 0]), np.log10(separatrix_right[:, 1]), 'r--', label='Separatrix Right')
    for fp in fixed_points:
        plt.plot(np.log10(fp[0]), np.log10(fp[1]), 'ko')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.plot(np.log10(sol_transient[:, 0]), np.log10(sol_transient[:, 1]), color='green', label='Transient')
    plt.xlabel('myofibroblasts, mF (cells per ml)')
    plt.ylabel('macrophages, M (cells per ml)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(np.log10(separatrix_left[:, 0]), np.log10(separatrix_left[:, 1]), 'r--', label='Separatrix Left')
    plt.plot(np.log10(separatrix_right[:, 0]), np.log10(separatrix_right[:, 1]), 'r--', label='Separatrix Right')
    for fp in fixed_points:
        plt.plot(np.log10(fp[0]), np.log10(fp[1]), 'ko')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.plot(np.log10(sol_repetitive[:, 0]), np.log10(sol_repetitive[:, 1]), color='red', label='Repetitive')
    plt.xlabel('myofibroblasts, mF (cells per ml)')
    plt.ylabel('macrophages, M (cells per ml)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(np.log10(separatrix_left[:, 0]), np.log10(separatrix_left[:, 1]), 'r--', label='Separatrix Left')
    plt.plot(np.log10(separatrix_right[:, 0]), np.log10(separatrix_right[:, 1]), 'r--', label='Separatrix Right')
    for fp in fixed_points:
        plt.plot(np.log10(fp[0]), np.log10(fp[1]), 'ko')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.plot(np.log10(sol_prolonged[:, 0]), np.log10(sol_prolonged[:, 1]), color='blue', label='Prolonged')
    plt.xlabel('myofibroblasts, mF (cells per ml)')
    plt.ylabel('macrophages, M (cells per ml)')
    plt.legend()

    plt.tight_layout()
    plt.show()
