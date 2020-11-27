#!/usr/bin/env python
# coding: utf-8
import pathlib
from copy import copy
from openpmd_viewer import addons
from matplotlib import pyplot as plt, colors, cm, rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import unyt as u
from prepic import lwfa
import numpy as np
from scipy.signal import hilbert
import custom_colormap as cc
import figformat

fig_width, fig_height, params = figformat.figure_format(fig_width=3.4)
rcParams.update(params)

cc.colormap_alpha('Reds')

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

p = pathlib.Path.cwd() / "simOutput" / "h5"
ts = addons.LpaDiagnostics(p)

rho, rho_info = ts.get_field(
    field="e_density",
    iteration=50000,
    slice_across="z",
)
electric, electric_info = ts.get_field(
    field="E",
    coord="z",
    iteration=50000,
    slice_across="z",
)
# get laser envelope
envelope, info = ts.get_laser_envelope(iteration=50000, pol='z', slice_across='z', laser_coord='y')

# get longitudinal field
e_y_of_y, e_y_of_y_info = ts.get_field(
    field="E",
    coord="y",
    iteration=50000,
    slice_across=["z", "x"],
)


fig, ax = plt.subplots(figsize=(fig_width, fig_height))

im_rho = ax.imshow(
    np.flipud(np.rot90(rho / n_c)),
    extent=np.roll(rho_info.imshow_extent * 1e6, 2),
    origin="lower",
    norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
    cmap=cm.get_cmap("cividis"),
)
im_envelope = ax.imshow(
    np.flipud(np.rot90(envelope / E0)),
    extent=np.roll(electric_info.imshow_extent * 1e6, 2),
    origin="lower",
    cmap='Reds_alpha',
)
im_envelope.set_clim(vmin=1.0)

# plot longitudinal field
ax.plot(e_y_of_y_info.y * 1e6, e_y_of_y / E0 * 25 + 10, color="tab:gray")
ax.axhline(10, color="tab:gray", ls="-.")

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
cbar_env.set_label(r"$eE_{z} / m c \omega_\mathrm{L}$")
cbar_rho.set_label(r"$n_{e} / n_\mathrm{cr}$")
cbar_rho.set_ticks([1e-4,1e-2,1e0])
# Add the name of the axes
ax.set_ylabel("$x \;(\mu \mathrm{m} )$")
ax.set_xlabel("$y \;(\mu \mathrm{m} )$")

fig.savefig(
    pathlib.Path.cwd()/"laser_density.png",
    dpi=600,
    transparent=False,
    bbox_inches="tight",
)
