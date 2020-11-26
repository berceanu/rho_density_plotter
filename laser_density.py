import pathlib
from copy import copy
from openpmd_viewer import addons
from matplotlib import pyplot as plt, colors, cm, rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import unyt as u
from prepic import lwfa
import numpy as np
from scipy.signal import hilbert
import colorcet as cc


my_cmap = copy(cc.m_fire)
my_cmap.set_under("k", alpha=0)

a0 = 2.4 * u.dimensionless  # Laser amplitude
tau = 25.0e-15 / 2.354820045 * u.second  # Laser duration
w0 = 22.0e-6 / 1.17741 * u.meter  # Laser waist
lambda0 = 0.8e-6 * u.meter  # Laser wavelength

laser = lwfa.Laser.from_a0(
    a0=a0,
    τL=tau,
    beam=lwfa.GaussianBeam(w0=w0, λL=lambda0),
)
n_c = laser.ncrit.to_value("1/m**3")
# 1.7419595910637713e+27
E0 = (laser.E0 / a0).to_value("volt/m")
# 4013376052599.5396
qe = u.qe.to_value("C")
# -1.6021766208e-19

p = pathlib.Path.cwd() / "diags" / "hdf5"
ts = addons.LpaDiagnostics(p)

# the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
# the initial density n_e has units of N/V, N = electron number
# multiply by electron charge -q_e to get (N e) / V
# so we get Q / N e, which is C/C, i.e. dimensionless
# Note: one can also normalize by the critical density n_c

rho, rho_info = ts.get_field(
    field="rho",
    iteration=40110,
    plot=True,
)
envelope, env_info = ts.get_laser_envelope(iteration=40110, pol='x')

# get longitudinal field
e_z_of_z, e_z_of_z_info = ts.get_field(
    field="E",
    coord="z",
    iteration=40110,
    slice_across="r",
)


fig, ax = plt.subplots(figsize=(20, 8))

im_rho = ax.imshow(
    rho / (np.abs(qe) * n_c),
    extent=rho_info.imshow_extent * 1e6,
    origin="lower",
    norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
    cmap=cm.get_cmap("cividis"),
)
im_envelope = ax.imshow(
    envelope / E0,
    extent=env_info.imshow_extent * 1e6,
    origin="lower",
    cmap=my_cmap,
)
im_envelope.set_clim(vmin=1.0)

# plot longitudinal field
ax.plot(e_z_of_z_info.z * 1e6, e_z_of_z / E0 * 25 - 20, color="tab:gray")
ax.axhline(-20, color="tab:gray", ls="-.")

cbaxes_rho = inset_axes(
    ax,
    width="3%",  # width = 10% of parent_bbox width
    height="46%",  # height : 50%
    loc=2,
    bbox_to_anchor=(1.01, 0.0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cbaxes_env = inset_axes(
    ax,
    width="3%",  # width = 5% of parent_bbox width
    height="46%",  # height : 50%
    loc=3,
    bbox_to_anchor=(1.01, 0.0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cbar_env = fig.colorbar(
    mappable=im_envelope, orientation="vertical", ticklocation="right", cax=cbaxes_env
)
cbar_rho = fig.colorbar(
    mappable=im_rho, orientation="vertical", ticklocation="right", cax=cbaxes_rho
)
cbar_env.set_label(r"$eE_{x} / m c \omega_\mathrm{L}$")
cbar_rho.set_label(r"$n_{e} / n_\mathrm{cr}$")
# cbar_rho.set_ticks([1e-4,1e-2,1e0])

# Add the name of the axes
ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))
ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))

fig.savefig(
    "laser_density.png",
    dpi=300,
    transparent=False,
    bbox_inches="tight",
)
