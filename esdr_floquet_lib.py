import numpy as np

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

    Delta_MW = D_GS - omega_MW
    Delta_b_MW = Delta_MW + M_x
    Delta_d_MW = Delta_MW - M_x

    H0 = np.array([
        [Delta_b_MW,  lambda_b,     0          ],
        [lambda_b,    0,            1j*lambda_d],
        [0,           -1j*lambda_d, Delta_d_MW ],
    ])
    H1 = np.array([
        [0,              0, Omega_RF_power],
        [0,              0, 0             ],
        [Omega_RF_power, 0, 0             ],
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
    assert alpha in ('0', 'B', 'D')
    assert beta in ('0', 'B', 'D')
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

def get_D_GS_eff(D_GS, M_x, B_x, B_y):
    """
    Effective zero-field splitting.
    """
    gamma_NV = (2*pi)*2.8025e10 # rad/(s T)
    term_x = np.square(gamma_NV*B_x)/(D_GS+M_x)
    term_y = np.square(gamma_NV*B_y)/(D_GS-M_x)
    D_GS_eff = D_GS + (3./2)*(term_x + term_y)
    return D_GS_eff

def get_M_x_eff(D_GS, M_x, B_x, B_y):
    """
    Effective strain along x-direction.
    """
    gamma_NV = (2*pi)*2.8025e10 # rad/(s T)
    term_x = np.square(gamma_NV*B_x)/(D_GS+M_x)
    term_y = np.square(gamma_NV*B_y)/(D_GS-M_x)
    M_x_eff = M_x + (1./2)*(term_x - term_y)
    return M_x_eff

def get_lamba_b_prime(lambda_b, lambda_d, omega_L, M_x_eff):
    if omega_L == M_x_eff == 0.0:
        return lambda_b
    V = np.hypot(M_x_eff, omega_L)
    numerator = (M_x_eff + V)*lambda_b - 1j*omega_L*lambda_d
    denominator = np.hypot(M_x_eff + V, omega_L)
    lambda_b_prime = numerator/denominator
    return lambda_b_prime

def get_lamba_d_prime(lambda_b, lambda_d, omega_L, M_x_eff):
    if omega_L == M_x_eff == 0.0:
        return lambda_d
    V = np.hypot(M_x_eff, omega_L)
    numerator = 1j*omega_L*lambda_b + (M_x_eff + V)*lambda_d
    denominator = np.hypot(M_x_eff + V, omega_L)
    lambda_d_prime = numerator/denominator
    return lambda_d_prime

def get_H_F_prime(
    n,
    D_GS_eff,
    M_x_eff,
    omega_MW,
    omega_RF,
    Omega_RF_power,
    omega_L,
    lambda_b = None,
    lambda_d = None,
    lambda_b_prime = None,
    lambda_d_prime = None,
):
    V = np.hypot(M_x_eff, omega_L)
    if lambda_b_prime is None and lambda_b is not None and lambda_d is not None:
        lambda_b_prime = get_lamba_b_prime(lambda_b, lambda_d, omega_L, M_x_eff)
    elif lambda_b_prime is not None:
        pass
    else:
        raise ValueError("must specify either lambda_b or lambda_b_prime")
    if lambda_d_prime is None and lambda_b is not None and lambda_d is not None:
        lambda_d_prime = get_lamba_d_prime(lambda_b, lambda_d, omega_L, M_x_eff)
    elif lambda_d_prime is not None:
        pass
    else:
        raise ValueError("must specify either lambda_d or lambda_d_prime")

    Delta_MW = D_GS_eff - omega_MW
    Delta_b_prime_no_omega_RF = Delta_MW + V
    Delta_d_prime_no_omega_RF = Delta_MW - V
    H0 = np.array([
        [Delta_b_prime_no_omega_RF,     lambda_b_prime,                   0],
        [np.conjugate(lambda_b_prime),  0,                                1j*lambda_d_prime],
        [0,                             np.conjugate(1j*lambda_d_prime),  Delta_d_prime_no_omega_RF],
    ], dtype=np.dtype('complex128'))
    H1 = np.array([
        [omega_L*Omega_RF_power/V, 0, M_x_eff*Omega_RF_power/V ],
        [0,                        0, 0                        ],
        [M_x_eff*Omega_RF_power/V, 0, -omega_L*Omega_RF_power/V],
    ], dtype=np.dtype('complex128'))
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
    I_omega_RF_N = N_omega_matrix(n, omega_RF_freq = omega_RF)
    H_F_prime = Fm_Hm + I_omega_RF_N
    return H_F_prime

class Params:
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'
    def __str__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'

class Results:
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'
    def __str__(self):
        return self.__class__.__name__ + '(' + str(list(self.__dict__.keys())) + ')'
def write_simulation_info_to_hdf5_file(params, results, filepath):
    import h5py
    with h5py.File(filepath, 'w') as fp:
        results_group = fp.create_group("simulation_results")
        for name, value in results.__dict__.items():
            try:
                if name in results.exclude:
                    continue
                elif name in results.compression:
                    results_group.create_dataset(name, data=value, compression=results.compression[name])
                else:
                    results_group.create_dataset(name, data=value)
            except TypeError:
                print("name = '{}'".format(name))
                raise
        param_group = fp.create_group("simulation_params")
        for name, value in params.__dict__.items():
            try:
                if name in params.exclude:
                    continue
                elif name in params.compression:
                    param_group.create_dataset(name, data=value, compression=params.compression[name])
                else:
                    param_group.create_dataset(name, data=value)
            except TypeError:
                print("name = '{}'".format(name))
                raise