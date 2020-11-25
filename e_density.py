import pathlib
from openpmd_viewer import OpenPMDTimeSeries
from matplotlib import pyplot, colors
import unyt as u
from prepic import lwfa
import numpy as np

a0 = 2.4  # Laser amplitude
tau = 25.0e-15 / 2.354820045  # Laser duration, sec
w0 = 22.0e-6 / 1.17741  # Laser waist
lambda0 = 0.8e-6  # Laser wavelength

laser = lwfa.Laser.from_a0(
    a0=a0 * u.dimensionless,
    τL=tau * u.second,
    beam=lwfa.GaussianBeam(w0=w0 * u.meter, λL=lambda0 * u.meter),
)
n_c = laser.ncrit.to_value("1/m**3")
# 1.7419595910637713e+27


p = pathlib.Path.cwd() / "betatron0007" / "simOutput" / "h5"
ts = OpenPMDTimeSeries(p)


rho, rho_info = ts.get_field(
    field="e_density",
    coord=None,
    iteration=50000,
    theta=None,
    slice_across="z",
)

fig, ax = pyplot.subplots(figsize=(7, 5))

# Plot the data
im = ax.imshow(
    np.rot90(rho / n_c),
    extent=np.roll(rho_info.imshow_extent * 1e6, 2),
    interpolation="nearest",
    aspect="auto",
    norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
)
# Add the name of the axes
ax.set_ylabel("$%s \;(\mu m)$" % rho_info.axes[1])
ax.set_xlabel("$%s \;(\mu m)$" % rho_info.axes[0])

fig.colorbar(im)
fig.savefig("e_density_z.png")
