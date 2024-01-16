import numpy as np
import datetime
import time
import h5py

# TODO: add a --dry-run parameter

t0 = time.perf_counter()
simulation_start = datetime.datetime.now()

pi = np.pi
MHz = 1e6
GHz = 1e9

def N_omega_matrix(n, omega_RF_freq, order_identity=3):
    ident = np.identity(order_identity)
    range_n = list(range(n, -(n+1), -1))
    order = ((2*n)+1)*order_identity
    N_omega = np.zeros((order, order))
    for i, this_n in enumerate(range_n):
        for j in range(order_identity):
            k = (i*order_identity) + j
            N_omega[k,k] = this_n*omega_RF_freq
    return N_omega
def get_Fm_Hm(
        n,
        D_GS,
        M_x,
        omega_MW,
        lambda_b,
        lambda_d,
        Omega_RF_power,
):
    ω_MW = omega_MW
    λb = lambda_b
    λd = lambda_d
    Ω_RF = Omega_RF_power

    Δ_MW = D_GS - ω_MW
    Δb_MW = Δ_MW + M_x
    Δd_MW = Δ_MW - M_x
    H0 = np.array([
        [Δb_MW, λb,     0],
        [λb,    0,      1j*λd],
        [0,     -1j*λd, Δd_MW],
    ])
    H1 = np.array([
        [0,    0, Ω_RF],
        [0,    0, 0],
        [Ω_RF, 0, 0],
    ])
    z3x3 = np.array(np.zeros((3, 3)))
    N = 2*n+1
    blocks = [ [ None for _ in range(N) ] for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            if i == j:
                blocks[j][i] = H0
            elif i == j+1 or i == j-1:
                blocks[j][i] = H1
            else:
                blocks[j][i] = z3x3
    Fm_Hm = np.block(blocks)
    return Fm_Hm

def ket_alpha_n(alpha, n, N):
    assert alpha in ('B', '0', 'D')
    n_range = list(range(-N, N+1, 1))
    assert n in n_range
    NV_state = {
        'B': [1,0,0],
        '0': [0,1,0],
        'D': [0,0,1],
    }
    length = 3*(2*N+1)
    rows = (N-n)*[0,0,0] + NV_state[alpha] + (N+n)*[0,0,0]
    column_vector = np.c_[rows].reshape(length)
    return column_vector
def bra_beta_n(beta, n, N):
    assert beta in ('B', '0', 'D')
    n_range = list(range(-N, N+1, 1))
    assert n in n_range
    NV_state = {
        'B': [1,0,0],
        '0': [0,1,0],
        'D': [0,0,1],
    }
    length = 3*(2*N+1)
    row_list = (N-n)*[0,0,0] + NV_state[beta] + (N+n)*[0,0,0]
    row_vector = np.array(row_list)
    return row_vector
def P_alpha_beta(alpha, beta, eigenvectors):
    # assert alpha in ('0', 'B', 'D')
    # assert beta in ('0', 'B', 'D')
    P_ab = 0.
    length, n_eigvecs = eigenvectors.shape
    N = (length-3)//6
    alpha_0_ket = ket_alpha_n(alpha, 0, N)
    for n in range(-N,N+1,1):
        this_beta_n_bra = bra_beta_n(beta, n, N)
        for i in range(n_eigvecs):
            eigenket = eigenvectors[:, i]
            eigenbra = eigenket.T
            P_ab += np.abs(np.vdot(this_beta_n_bra, eigenket)**2 * np.vdot(eigenbra, alpha_0_ket)**2)
    return P_ab

N = 4
MW_freqs = np.linspace(2.867*GHz, 2.897*GHz, 400+1)
RF_freqs = np.linspace(0.0*MHz, 12.0*MHz, 160+1)
P_0_B = np.zeros((len(RF_freqs), len(MW_freqs)))
P_0_D = np.zeros((len(RF_freqs), len(MW_freqs)))
P_0_0 = np.zeros((len(RF_freqs), len(MW_freqs)))
H = np.empty((len(RF_freqs), len(MW_freqs), 6*N+3, 6*N+3), dtype=np.dtype('complex128'))
eigvals = np.empty((len(RF_freqs), len(MW_freqs), 6*N+3, 6*N+3), dtype=np.dtype('float64'))
eigvecs = np.empty((len(RF_freqs), len(MW_freqs), 6*N+3, 6*N+3), dtype=np.dtype('complex128'))
D_GS = 2.8825*GHz*(2*pi)
M_x = 0.5*9.09*MHz*(2*pi)
lambda_b = 0.12*MHz*(2*pi)
lambda_d = 0.12*MHz*(2*pi)
Omega_RF_power = 0.98*MHz*(2*pi)
t1 = time.perf_counter()
for i, RF_freq in enumerate(RF_freqs):
    for j, MW_freq in enumerate(MW_freqs):
        H[i][j] = N_omega_matrix(N, omega_RF_freq = RF_freq*(2*pi)) + get_Fm_Hm(
                N,
                D_GS = D_GS,
                M_x = M_x,
                omega_MW = MW_freq*(2*pi),
                lambda_b = lambda_b,
                lambda_d = lambda_d,
                Omega_RF_power = Omega_RF_power,
        )
t2 = time.perf_counter()
for i, RF_freq in enumerate(RF_freqs):
    for j, MW_freq in enumerate(MW_freqs):
        eigvals[i][j], eigvecs[i][j] = np.linalg.eigh(H[i][j]);
t3 = time.perf_counter()
for i, RF_freq in enumerate(RF_freqs):
    for j, MW_freq in enumerate(MW_freqs):
        P_0_B[i][j] = P_alpha_beta('0', 'B', eigvecs[i][j])
        P_0_D[i][j] = P_alpha_beta('0', 'D', eigvecs[i][j])
        P_0_0[i][j] = P_alpha_beta('0', '0', eigvecs[i][j])
t4 = time.perf_counter()
class SimulationParams:
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'
    def __str__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'

params = SimulationParams()
params.MW_freqs = MW_freqs
params.RF_freqs = RF_freqs
params.D_GS = D_GS
params.M_x = M_x
params.lambda_b = lambda_b
params.lambda_d = lambda_d
params.Omega_RF_power = Omega_RF_power
params.N = N
np.savez_compressed(simulation_start.strftime("%Y-%m-%d_%H_%M_%S") + '_P.npz', P_0_B=P_0_B, P_0_D=P_0_D, P_0_0=P_0_0)

t5 = time.perf_counter()

print("preamble = {}".format(t1-t0))
print("initialize hamiltonians = {}".format(t2-t1))
print("diagonalize hamiltonians = {}".format(t3-t2))
print("find P0 = {}".format(t4-t3))
print("postamble = {}".format(t5-t4))
print("total = {}".format(t5-t0))

