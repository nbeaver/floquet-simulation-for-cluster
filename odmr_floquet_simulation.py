import numpy as np
import datetime
import time
import h5py
import esdr_floquet_lib

t0 = time.perf_counter()
simulation_start = datetime.datetime.now()

pi = np.pi
MHz = 1e6
GHz = 1e9

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
        H[i][j] = esdr_floquet_lib.N_omega_matrix(N, omega_RF_freq = RF_freq*(2*pi)) + esdr_floquet_lib.get_Fm_Hm(
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
        P_0_B[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'B', eigvecs[i][j])
t4 = time.perf_counter()
for i, RF_freq in enumerate(RF_freqs):
    for j, MW_freq in enumerate(MW_freqs):
        P_0_D[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'D', eigvecs[i][j])
t5 = time.perf_counter()
for i, RF_freq in enumerate(RF_freqs):
    for j, MW_freq in enumerate(MW_freqs):
        P_0_0[i][j] = esdr_floquet_lib.P_alpha_beta('0', '0', eigvecs[i][j])
t6 = time.perf_counter()
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

t7 = time.perf_counter()

print("preamble = {}".format(t1-t0))
print("initialize hamiltonians = {}".format(t2-t1))
print("diagonalize hamiltonians = {}".format(t3-t2))
print("find P_0_B = {}".format(t4-t3))
print("find P_0_D = {}".format(t5-t4))
print("find P_0_0 = {}".format(t6-t5))
print("postamble = {}".format(t7-t6))
print("total = {}".format(t7-t0))

