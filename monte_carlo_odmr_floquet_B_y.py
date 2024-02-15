import esdr_floquet_lib
import numpy as np
import secrets
import time
import datetime
import argparse
import os.path
import logging

logger = logging.getLogger(__name__)

def get_Floquet_Hamiltonian_shape(arr1, arr2, N):
    shape = (len(arr1), len(arr2), 6*N+3, 6*N+3)
    return shape
def get_transition_probability_shape(arr1, arr2):
    shape = (len(arr1), len(arr2))
    return shape

def get_random(mean, stdev, shape=None, rng=None):
    if rng is None:
        import secrets
        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)
    random = rng.normal(size=shape, loc=mean, scale=stdev)
    return random

def get_params():
    from math import pi
    MHz = 1e6
    GHz = 1e9
    gauss = 1e-4 # T
    p = esdr_floquet_lib.Params()
    p.gamma_NV = (2*pi)*2.8025e10 # rad/(s T)
    p.B_x = 3e-4 # tesla
    # p.B_y = 3e-4 # tesla
    p.B_z = 5e-4 # tesla
    p.M_x = 2.0*MHz*(2*pi)
    p.N = 4

    p.MW_step = 0.1*MHz

    p.D_GS = 2.87*GHz*(2*pi)
#     p.Omega_RF_power = 0.98*MHz*(2*pi)
    p.Omega_RF_power = 0.0 # No RF
    p.omega_RF = 0.0 # Zero frequency
    p.lambda_b = 0.12*MHz*(2*pi)
    p.lambda_d = 0.12*MHz*(2*pi)

    # Monte Carlo parameters.
    p.N_avg = 300
    p.mu_B_y = 0.0*gauss
    p.sigma_B_y = 50*gauss
    return p

def setup_params(params):
    from math import pi
    MHz = 1e6
    GHz = 1e9
    seed = secrets.randbits(128)
    params.random_seed = str(seed)
    rng = np.random.default_rng(seed)
    params.B_y = get_random(mean=params.mu_B_y, stdev=params.sigma_B_y, shape=params.N_avg, rng=rng)

    params.omega_L = params.gamma_NV*params.B_z
    params.MW_start_freq = 2.87*GHz - np.abs(params.omega_L/(2*pi)) - 15*MHz
    params.MW_stop_freq  = 2.87*GHz + np.abs(params.omega_L/(2*pi)) + 15*MHz
    params.MW_range = params.MW_stop_freq - params.MW_start_freq
    params.MW_N_steps = round(params.MW_range/params.MW_step)+1
    # TODO: also account for RF splitting
    params.MW_freqs = np.linspace(params.MW_start_freq, params.MW_stop_freq, params.MW_N_steps)
    params.omega_MWs = params.MW_freqs*2*pi

    params.D_GS_eff = esdr_floquet_lib.get_D_GS_eff(params.D_GS, params.M_x, params.B_x, params.B_y)
    params.M_x_eff = esdr_floquet_lib.get_M_x_eff(params.D_GS, params.M_x, params.B_x, params.B_y)
    params.lambda_b_prime = esdr_floquet_lib.get_lambda_b_prime(
        params.lambda_b, params.lambda_b, params.omega_L, params.M_x_eff)
    params.lambda_d_prime = esdr_floquet_lib.get_lambda_d_prime(
        params.lambda_b, params.lambda_b, params.omega_L, params.M_x_eff)
def do_simulation(params):
    arr1 = params.B_y
    arr2 = params.omega_MWs
    H_shape = get_Floquet_Hamiltonian_shape(arr1, arr2, params.N)

    results = esdr_floquet_lib.Results()
    results.H = np.empty(H_shape, dtype=np.dtype('complex128'))
    results.eigvals = np.empty(H_shape[:-1], dtype=np.dtype('float64'))
    results.eigvecs = np.empty(H_shape, dtype=np.dtype('complex128'))

    results.P_0_B = np.empty(H_shape[:-2])
    results.P_0_D = np.empty(H_shape[:-2])
    results.P_0_0 = np.empty(H_shape[:-2])

    env_vars = [
        'SLURM_JOB_START_TIME',
        'SLURM_JOB_NAME',
        'SLURM_MEM_PER_CPU',
        'SLURM_JOB_ID',
        'SLURM_JOB_USER',
        'SLURM_SUBMIT_DIR',
        'SLURM_JOB_ACCOUNT'
    ]
    for env_var in env_vars:
        try:
            setattr(results, env_var, os.environ[env_var])
        except KeyError:
            pass

    date_start = datetime.datetime.now()
    t_start = time.perf_counter()
    for i, B_y_i in enumerate(arr1):
        for j, omega_MW in enumerate(arr2):
            results.H[i][j] = esdr_floquet_lib.get_H_F_prime(
                    n = params.N,
                    D_GS_eff = params.D_GS_eff[i],
                    M_x_eff = params.M_x_eff[i],
                    omega_RF = params.omega_RF,
                    omega_MW = omega_MW,
                    Omega_RF_power = params.Omega_RF_power,
                    omega_L = params.omega_L,
                    lambda_b_prime = params.lambda_b_prime[i],
                    lambda_d_prime = params.lambda_d_prime[i],
            )
            results.eigvals[i][j], results.eigvecs[i][j] = np.linalg.eigh(results.H[i][j])
            results.P_0_B[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'B', results.eigvecs[i][j])
            results.P_0_D[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'D', results.eigvecs[i][j])
            results.P_0_0[i][j] = esdr_floquet_lib.P_alpha_beta('0', '0', results.eigvecs[i][j])
    t_stop = time.perf_counter()
    date_stop = datetime.datetime.now()
    del results.H
    del results.eigvecs
    del results.eigvals

    results.duration_s = t_stop - t_start
    results.date_start_iso = date_start.isoformat()
    results.date_stop_iso = date_stop.isoformat()
    results.date_start_ctime = date_start.ctime()
    results.date_stop_ctime = date_stop.ctime()
    results.date_start_locale_time = date_start.strftime("%c")
    results.date_stop_locale_time = date_stop.strftime("%c")
    results.date_start_unix = time.mktime(date_start.timetuple())
    results.date_stop_unix = time.mktime(date_stop.timetuple())

    results.P_0_0_avg = np.mean(results.P_0_0, axis=0)
    results.P_0_B_avg = np.mean(results.P_0_B, axis=0)
    results.P_0_D_avg = np.mean(results.P_0_D, axis=0)

    results.P_0_0_std = np.std(results.P_0_0, axis=0)
    results.P_0_B_std = np.std(results.P_0_B, axis=0)
    results.P_0_D_std = np.std(results.P_0_D, axis=0)

    results.compression = {
        'P_0_0': 'lzf',
        'P_0_B': 'lzf',
        'P_0_D': 'lzf',
        'P_0_0_avg': 'lzf',
        'P_0_B_avg': 'lzf',
        'P_0_D_avg': 'lzf',
        'P_0_0_std': 'lzf',
        'P_0_B_std': 'lzf',
        'P_0_D_std': 'lzf',
    }
    results.exclude = [
        'compression',
        'exclude',
        'H',
        'eigvals',
        'eigvecs',
    ]
    params.compression = {
        'MW_freqs': 'lzf',
        'omega_MWs': 'lzf',
        'B_y': 'lzf',
        'D_GS_eff': 'lzf',
        'M_x_eff': 'lzf',
        'lambda_b_prime': 'lzf',
        'lambda_d_prime': 'lzf',
    }
    params.exclude = ['compression', 'exclude']

    return params, results
def main():
    gauss = 1e-4  # T
    parser = argparse.ArgumentParser(
        description='ODMR simulation via Floquet, B_y Monte Carlo')
    parser.add_argument(
        '--n-avg',
        type=int,
        help='number of averages')
    parser.add_argument(
        '--param-start',
        type=float,
        default=0.0*gauss,
        help='parameter sweep start value')
    parser.add_argument(
        '--param-stop',
        type=float,
        default=20*gauss,
        help='parameter sweep stop value')
    parser.add_argument(
        '--param-steps',
        type=int,
        default=50,
        help='parameter sweep number of steps')
    parser.add_argument(
        '--tag-filename',
        default='',
        help='tag to add to filename')
    parser.add_argument(
        '--out-dir',
        default='.',
        help='output directory')
    parser.add_argument(
        '-v',
        '--verbose',
        help='More verbose logging',
        dest="loglevel",
        default=logging.WARNING,
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='Enable debugging logs',
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    logger.setLevel(args.loglevel)
    outdir = args.out_dir

    start = args.param_start
    stop = args.param_stop
    n_steps = args.param_steps
    for i, sigma_B_y in enumerate(np.linspace(start, stop, n_steps)):
        logging.info("{} of {}".format(i+1, n_steps)) # crude progress meter
        params = get_params()
        params.sigma_B_y = sigma_B_y
        if args.n_avg is not None:
            params.N_avg = args.n_avg
        setup_params(params)
        params, results = do_simulation(params)
        filename = "odmr_floquet_monte_carlo_B_y_{}_{:04d}.hdf5".format(args.tag_filename, i)
        parent_dir = os.path.join(outdir, "full")
        os.makedirs(parent_dir, exist_ok=True)
        filepath = os.path.join(parent_dir, filename)
        esdr_floquet_lib.write_simulation_info_to_hdf5_file(
            params,
            results,
            filepath=filepath
        )
        # Save data sets that only have the averages and so are ~1/N_avg smaller.
        del results.P_0_0
        del results.P_0_B
        del results.P_0_D
        del filename, parent_dir, filepath # avoid re-using these variables
        filename = "odmr_floquet_monte_carlo_B_y_{}_avg_{:04d}.hdf5".format(args.tag_filename, i)
        parent_dir = os.path.join(outdir, "avg")
        os.makedirs(parent_dir, exist_ok=True)
        filepath = os.path.join(parent_dir, filename)
        esdr_floquet_lib.write_simulation_info_to_hdf5_file(
            params,
            results,
            filepath=filepath
        )
        del filename, parent_dir, filepath  # avoid re-using these variables

if __name__ == '__main__':
    main()
