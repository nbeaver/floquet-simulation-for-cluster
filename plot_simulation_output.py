import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(constrained_layout=True, figsize=(8,6))
plot = ax.pcolormesh(
    MW_freqs/GHz,
    RF_freqs/MHz,
    P_0_B,
    cmap='magma',
)
fig.colorbar(mappable=plot, ax=ax);
fig.suptitle("$P_{\\left|0\\right> \\rightarrow \\left|B\\right>}$"
     + ", Floquet order = {}".format(N),
     fontsize=18
);
ax.set_ylabel("RF Frequency [MHz]")
ax.set_xlabel("MW Frequency [GHz]")
ax.axhline(y=5.50, color='white', label="5.50 MHz", linestyle='--');
ax.text(x=MW_freqs[0]/GHz, y=5.50+0.1, s="5.50 MHz", color="white");
ax.axhline(y=9.09, color='white', label="9.09 MHz", linestyle='--');
ax.text(x=MW_freqs[0]/GHz, y=9.09+0.1, s="9.09 MHz", color="white");
ax.set_title(
    "$\\tilde{{D}}_{{GS}}/2\\pi$ = {} GHz".format(D_GS/(2*pi*GHz))
    + ", $\\tilde{{M}}_{{x}}/2\\pi$ = {} MHz".format(M_x/(2*pi*MHz))
    + ", $\\lambda^b$ = {:.3f} MHz".format(lambda_b/(2*pi*MHz))
    + ", $\\lambda^d$ = {:.3f} MHz".format(lambda_d/(2*pi*MHz))
    + ", $\\Omega_{{RF}}$ = {:.3f} MHz".format(Omega_RF_power/(2*pi*MHz)),
    fontsize=8,
);