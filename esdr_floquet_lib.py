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

def get_lamba_b_prime(lambda_b, lambda_d, omega_L, M_x):
    V = np.hypot(M_x, omega_L)
    numerator = (M_x + V)*lambda_b - 1j*omega_L*lambda_d
    denominator = np.hypot(M_x+V, omega_L)
    lambda_b_prime = numerator/denominator
    return lambda_b_prime

def get_lamba_d_prime(lambda_b, lambda_d, omega_L, M_x):
    V = np.hypot(M_x, omega_L)
    numerator = 1j*omega_L*lambda_b + (M_x + V)*lambda_d
    denominator = np.hypot(M_x+V, omega_L)
    lambda_d_prime = numerator/denominator
    return lambda_d_prime

def get_H_F_prime(
    n,
    D_GS,
    M_x,
    omega_MW,
    omega_RF,
    Omega_RF_power,
    omega_L,
    lambda_b = None,
    lambda_d = None,
    lambda_b_prime = None,
    lambda_d_prime = None,
):
    V = np.hypot(M_x, omega_L)
    if lambda_b_prime is None and lambda_b is not None and lambda_d is not None:
        lambda_b_prime = get_lamba_b_prime(lambda_b, lambda_d, omega_L, M_x)
    elif lambda_b_prime is not None:
        pass
    else:
        raise ValueError("must specify either lambda_b or lambda_b_prime")
    if lambda_d_prime is None and lambda_b is not None and lambda_d is not None:
        lambda_d_prime = get_lamba_d_prime(lambda_b, lambda_d, omega_L, M_x)
    elif lambda_d_prime is not None:
        pass
    else:
        raise ValueError("must specify either lambda_d or lambda_d_prime")

    Delta_MW = D_GS - omega_MW
    Delta_b_prime_no_omega_RF = Delta_MW + V
    Delta_d_prime_no_omega_RF = Delta_MW - V
    H0 = np.array([
        [Delta_b_prime_no_omega_RF,     lambda_b_prime,                   0],
        [np.conjugate(lambda_b_prime),  0,                                1j*lambda_d_prime],
        [0,                             np.conjugate(1j*lambda_d_prime),  Delta_d_prime_no_omega_RF],
    ])
    H1 = np.array([
        [omega_L*Omega_RF_power/V, 0, M_x*Omega_RF_power/V     ],
        [0,                        0, 0                        ],
        [M_x*Omega_RF_power/V,     0, -omega_L*Omega_RF_power/V],
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
    I_omega_RF_N = N_omega_matrix(n, omega_RF_freq = omega_RF)
    H_F_prime = Fm_Hm + I_omega_RF_N
    return H_F_prime
