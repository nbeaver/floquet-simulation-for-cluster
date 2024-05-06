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
    B_random = rng.normal(size=shape, loc=mean, scale=stdev)
    return B_random

def get_params():
    from math import pi
    MHz = 1e6
    GHz = 1e9
    gauss = 1e-4 # T
    p = esdr_floquet_lib.Params()
    p.gamma_NV = gamma_NV = (2*pi)*2.8025e10 # rad/(s T)
    p.B_x = 0*gauss
    p.B_y = 0*gauss
    # p.B_z = 5*gauss
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
    p.mu_B_z = 5.0*gauss
    p.sigma_B_z = 5*gauss
    return p

def setup_params(params):
    import math
    from math import pi
    MHz = 1e6
    GHz = 1e9
    seed = secrets.randbits(128)
    params.random_seed = str(seed)
    rng = np.random.default_rng(seed)

    params.D_GS_eff = esdr_floquet_lib.get_D_GS_eff(params.D_GS, params.M_x, params.B_x, params.B_y)
    params.M_x_eff = esdr_floquet_lib.get_M_x_eff(params.D_GS, params.M_x, params.B_x, params.B_y)

    B_z_random = get_random(mean=params.mu_B_z, stdev=params.sigma_B_z, shape=params.N_avg, rng=rng)
    params.B_z = B_z_random
    # We need consistent values for MW_start_freq and MW_stop_freq
    # to enable averaging, but since these values are randomly chosen
    # we don't know in advance what the largest will be.
    # So we use a value of mu + n*sigma, which should capture almost all values
    # (missing perhaps 1 in 15787 for 4 sigma).
    n_sigma = 4
    B_z_max = params.mu_B_z + n_sigma*params.sigma_B_z

    params.omega_L = params.gamma_NV*params.B_z
    omega_L_max = params.gamma_NV*B_z_max
    V = np.hypot(omega_L_max, params.M_x_eff)
    # Estimate shift based on resonant frequencies.
    shift = params.omega_RF/2. + np.hypot(V - params.omega_RF/2., params.Omega_RF_power)
    shift_Hz = shift/(2*pi)
    # Add on an extra 15 MHz to allow for peak width and bin to nearest step size.
    params.MW_start_freq = params.MW_step * math.floor(((params.D_GS_eff/(2*pi)) - shift_Hz - 15*MHz) / params.MW_step)
    params.MW_stop_freq  = params.MW_step * math.ceil(((params.D_GS_eff/(2*pi)) + shift_Hz + 15*MHz) / params.MW_step)

    params.MW_range = params.MW_stop_freq - params.MW_start_freq
    params.MW_N_steps = round(params.MW_range/params.MW_step)+1
    # TODO: also account for RF splitting
    params.MW_freqs = np.linspace(params.MW_start_freq, params.MW_stop_freq, params.MW_N_steps)
    params.omega_MWs = params.MW_freqs*2*pi

    params.lambda_b_prime = esdr_floquet_lib.get_lambda_b_prime(
        params.lambda_b, params.lambda_b, params.omega_L, params.M_x_eff)
    params.lambda_d_prime = esdr_floquet_lib.get_lambda_d_prime(
        params.lambda_b, params.lambda_b, params.omega_L, params.M_x_eff)
def do_simulation(params):
    arr1 = params.B_z
    arr2 = params.omega_MWs
    H_shape = get_Floquet_Hamiltonian_shape(arr1, arr2, params.N)

    results = esdr_floquet_lib.Results()
    results.H = np.empty(H_shape, dtype=np.dtype('complex128'))
    results.eigvals = np.empty(H_shape[:-1], dtype=np.dtype('float64'))
    results.eigvecs = np.empty(H_shape, dtype=np.dtype('complex128'))

    results.P_0_B_raw = np.empty(H_shape[:-2])
    results.P_0_D_raw = np.empty(H_shape[:-2])
    results.P_0_0_raw = np.empty(H_shape[:-2])

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
    for i, B_z_i in enumerate(params.B_z):
        for j, omega_MW in enumerate(params.omega_MWs):
            results.H[i][j] = esdr_floquet_lib.get_H_F_prime(
                    n = params.N,
                    D_GS_eff = params.D_GS_eff,
                    M_x_eff = params.M_x_eff,
                    omega_RF = params.omega_RF,
                    omega_MW = omega_MW,
                    Omega_RF_power = params.Omega_RF_power,
                    omega_L = params.omega_L[i],
                    lambda_b_prime = params.lambda_b_prime[i],
                    lambda_d_prime = params.lambda_d_prime[i],
            )
            results.eigvals[i][j], results.eigvecs[i][j] = np.linalg.eigh(results.H[i][j])
            results.P_0_B_raw[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'B', results.eigvecs[i][j])
            results.P_0_D_raw[i][j] = esdr_floquet_lib.P_alpha_beta('0', 'D', results.eigvecs[i][j])
            results.P_0_0_raw[i][j] = esdr_floquet_lib.P_alpha_beta('0', '0', results.eigvecs[i][j])
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

    # Remove any sweeps that contain NaNs since these cannot be averaged.
    results.P_0_0 = results.P_0_0_raw[~np.isnan(results.P_0_0_raw).any(axis=1)]
    results.P_0_B = results.P_0_B_raw[~np.isnan(results.P_0_B_raw).any(axis=1)]
    results.P_0_D = results.P_0_D_raw[~np.isnan(results.P_0_D_raw).any(axis=1)]

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
        'P_0_0_raw',
        'P_0_B_raw',
        'P_0_D_raw',
    ]
    params.compression = {
        'MW_freqs': 'lzf',
        'omega_MWs': 'lzf',
        'B_z': 'lzf',
        'omega_L': 'lzf',
        'lambda_b_prime': 'lzf',
        'lambda_d_prime': 'lzf',
    }
    params.exclude = ['compression', 'exclude']

    return params, results
def main():
    gauss = 1e-4  # T
    GHz = 1e9
    MHz = 1e6
    kHz = 1e3
    pi = np.pi
    # TODO: add flag for MW_step
    parser = argparse.ArgumentParser(
        description='ODMR simulation via Floquet, B_z Monte Carlo')
    parser.add_argument(
        '--n-avg',
        type=int,
        help='number of averages')
    parser.add_argument(
        '--mu-Bz',
        type=str,
        default=None,
        help='mu_Bz [T]')
    parser.add_argument(
        '--Mx',
        type=str,
        default=None,
        help='M_x [rad/s]')
    parser.add_argument(
        '--Bx',
        type=str,
        default=None,
        help='B_x [T]')
    parser.add_argument(
        '--By',
        type=str,
        default=None,
        help='B_y [T]')
    parser.add_argument(
        '--Dgs',
        type=str,
        default=None,
        help='D_gs [rad/s]')
    parser.add_argument(
        '--omega-rf-power',
        type=str,
        default=None,
        help='RF power [rad/s]')
    parser.add_argument(
        '--omega-rf',
        type=str,
        default=None,
        help='RF frequency [rad/s]')
    parser.add_argument(
        '--MW-step',
        type=str,
        default=None,
        help='MW step [Hz]')
    parser.add_argument(
        '--param-start',
        type=str,
        default='0.0e-4',
        help='parameter sweep start value')
    parser.add_argument(
        '--param-stop',
        type=str,
        default='5e-4',
        help='parameter sweep stop value')
    parser.add_argument(
        '--param-steps',
        type=int,
        default=51,
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

    start = float(eval(args.param_start))
    stop = float(eval(args.param_stop))
    n_steps = args.param_steps
    for i, sigma_B_z in enumerate(np.linspace(start, stop, n_steps)):
        logging.info("{} of {}".format(i+1, n_steps)) # crude progress meter
        params = get_params()
        params.sigma_B_z = sigma_B_z
        if args.n_avg is not None:
            params.N_avg = args.n_avg
        if args.mu_Bz is not None:
            params.mu_B_z = float(eval(args.mu_Bz))
        if args.Mx is not None:
            params.M_x = float(eval(args.Mx))
        if args.Bx is not None:
            params.B_x = float(eval(args.Bx))
        if args.By is not None:
            params.B_y = float(eval(args.By))
        if args.Dgs is not None:
            params.D_GS = float(eval(args.Dgs))
        if args.omega_rf_power is not None:
            params.Omega_RF_power = float(eval(args.omega_rf_power))
        if args.omega_rf is not None:
            params.omega_RF = float(eval(args.omega_rf))
        if args.MW_step is not None:
            params.MW_step = float(eval(args.MW_step))
        setup_params(params)
        params, results = do_simulation(params)
        filename = "odmr_floquet_monte_carlo_B_z_{}_{:04d}.hdf5".format(args.tag_filename, i)
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
        filename = "odmr_floquet_monte_carlo_B_z_{}_avg_{:04d}.hdf5".format(args.tag_filename, i)
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
