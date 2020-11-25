import pathlib
from copy import copy
from openpmd_viewer import addons
from matplotlib import pyplot, colors, cm
import unyt as u
from prepic import lwfa
import numpy as np
from scipy.signal import hilbert

my_cmap = copy(cm.get_cmap("inferno"))
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
# FIXME use ts.get_laser_envelope() once https://github.com/openPMD/openPMD-viewer/issues/292 is solved
e_complx = hilbert(electric, axis=0)
envelope = np.abs(e_complx)


fig, ax = pyplot.subplots(figsize=(7, 5))

im_rho = ax.imshow(
    np.flipud(np.rot90(rho / n_c)),
    extent=np.roll(rho_info.imshow_extent * 1e6, 2),
    interpolation="nearest",
    origin="lower",
    aspect="auto",
    norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
)
im_envelope = ax.imshow(
    np.flipud(np.rot90(envelope / E0)),
    extent=np.roll(electric_info.imshow_extent * 1e6, 2),
    interpolation="nearest",
    origin="lower",
    aspect="auto",
    cmap=my_cmap,
)
im_envelope.set_clim(vmin=1.0)
fig.colorbar(mappable=im_envelope)
fig.colorbar(mappable=im_rho)

# Add the name of the axes
ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))
ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))

fig.savefig("both.png")
