import h5py
import numpy as np
from scipy.signal import hilbert
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import figformat
from matplotlib.colors import LinearSegmentedColormap
import pwd 
import os

for n in ["Reds", "Greens", "Blues"]:
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(f"{n}")(range(ncolors))
    # change alpha values
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=f"{n}_alpha", colors=color_array
    )
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

fig_width, fig_height, params = figformat.figure_format(fig_width=3.4)
mpl.rcParams.update(params)


def get_snapshot(
    filename,
    iteration,
    species,
    simDim,
    movingWin,
    cmap1,
    field,
    E_log,
    cmap2,
    AMP,
    save,
    slice=None,
):
    """
    Plot Density and Ex or Ey in one picture.
        Parameters
        ----------
        filename : string
            filename to the directory

        iteration : int
            The iteration at which to obtain the data

        species : string
            Particle species to use for calculations

        simDim : string
            Dimension of the simulation: '2D' or '3D'

        movingWin : bool
            Whether to use fix or moving y-axis

        cmap1 : string
            colormap for density: 'Greys'

        field : bool
            Whether to plot field. If False, E_log,cmap2
            AMP are not relevant anymore but need to be
            included with any input.

        E_log : bool
            Whether to plot the lineout of longitudinal
            E-field

        cmap2 : string
            Colormap for E-field: 'hot_r'

        AMP : int
            Amplification factor for longitudinal E-field

        save : bool
            Whether to save the figure.
            Save figure in pdf and png format
    """
    # Get HDF5 files
    h5_dir = get_h5path(filename)
    h5f = h5py.File(rf"/{h5_dir}/simData_" + str(iteration) + ".h5", "r")
    time = get_time(filename, iteration=iteration)
    print(f"h5_dir: {h5_dir}")
    print("time = " + str(time))

    # Read Density data
    Den = h5f[f"data/{iteration}/fields/" + str(species) + "_density"]
    # Setup grid
    if simDim == "2D":
        iteration_x = Den.attrs["gridSpacing"][1] * Den.attrs["gridUnitSI"] * 1e6
        iteration_y = Den.attrs["gridSpacing"][0] * Den.attrs["gridUnitSI"] * 1e6
        grid_offset = Den.attrs["gridGlobalOffset"][0] * Den.attrs["gridUnitSI"] * 1e6
    elif simDim == "3D":
        iteration_x = Den.attrs["gridSpacing"][0] * Den.attrs["gridUnitSI"] * 1e6
        iteration_y = Den.attrs["gridSpacing"][1] * Den.attrs["gridUnitSI"] * 1e6
        grid_offset = Den.attrs["gridGlobalOffset"][1] * Den.attrs["gridUnitSI"] * 1e6

    n_points_x = Den.attrs["_global_size"][0]
    n_points_y = Den.attrs["_global_size"][1]

    start_x = Den.attrs["position"][0] * iteration_x
    end_x = start_x + (n_points_x - 1) * iteration_x

    if movingWin:
        start_y = grid_offset
        end_y = grid_offset + (n_points_y - 1) * iteration_y
    else:
        start_y = Den.attrs["position"][1] * iteration_y
        end_y = (n_points_y - 1) * iteration_y

    x = np.linspace(start_x, end_x, n_points_x, endpoint=True)
    y = np.linspace(start_y, end_y, n_points_y, endpoint=True)

    X, Y = np.meshgrid(y, x)
    extent = np.min(X), np.max(X), np.min(Y), np.max(Y)

    if simDim == "2D":
        Den = Den[:, :].T * Den.attrs["unitSI"]
        print("2D simulation snapshot at =" + str(time))
    elif simDim == "3D":
        slice_z = int(Den.attrs["_global_size"][2] // 2)
        Den = Den[:, :, slice_z] * Den.attrs["unitSI"]
        print("3D simulation snapshot at =" + str(time))
        print("slice at z =", int(slice_z))
    print(np.max(Den))
    Den = np.abs(Den / denunit)

    if field:
        # If Density and field are plotted, colorbar height is
        # 46% each
        colorbar_h = "46%"
        # Read fields data
        E_x = h5f[f"data/{iteration}/fields/E/x"]
        E_y = h5f[f"data/{iteration}/fields/E/y"]
        E_z = h5f[f"data/{iteration}/fields/E/z"]
        # Grid of slice at x
        slice_x = int(E_x.attrs["_global_size"][0] // 2)
        if simDim == "2D":
            E_x = E_x[:, :].T * E_x.attrs["unitSI"]
            if E_log:
                print("One scale for Ey correspond to " + str(1 / AMP) + "a0")
                E_y = E_y[:, slice_x] * E_y.attrs["unitSI"]
                E_y = E_y / exunit
        elif simDim == "3D":
            # Grid of slice z
            slice_z = int(E_z.attrs["_global_size"][2] // 2)
            if slice == None:
                E = E_x[:, :, slice_z] * E_x.attrs["unitSI"]
            elif slice == "z":
                E = E_z[:, :, slice_z] * E_x.attrs["unitSI"]
            if E_log:
                print("One scale for Ey correspond to " + str(1 / AMP) + "a0")
                E_y = E_y[slice_x, :, slice_z] * E_y.attrs["unitSI"]
                E_y = E_y / exunit

        # Make laser envolope
        E = E / exunit
        E = hilbert(E)
        E = np.abs(E)
        E_max = np.max(E)
        E_max = int(round(E_max))
        E_min = np.min(E)
        E_min = int(round(E_min))
    else:
        # If only Density is plot the colorbar height is 100%
        colorbar_h = "100%"

    # Plot density
    ax = plt.subplot()

    img2 = ax.imshow(
        Den,
        norm=LogNorm(vmin=1e-3),
        alpha=1,
        cmap=cmap1,
        extent=extent,
        interpolation="bilinear",
        aspect="auto",
    )
    cbaxes2 = inset_axes(
        ax,
        width="3%",  # width = 10% of parent_bbox width
        height=colorbar_h,  # height : 50%
        loc=2,
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cbar = plt.colorbar(img2, orientation="vertical", ticklocation="right", cax=cbaxes2)
    cbar.set_label(rf"$n_{species}$" "$/n_\mathrm{cr}$ ")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.minorticks_on()
    # Plot fields
    if field:
        img = ax.imshow(E, vmin=1, cmap=cmap2, extent=extent, aspect="auto")
        cbaxes = inset_axes(
            ax,
            width="3%",  # width = 5% of parent_bbox width
            height=colorbar_h,  # height : 50%
            loc=3,
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        cbar = plt.colorbar(
            img, orientation="vertical", ticklocation="right", cax=cbaxes
        )
        if slice == None:
            cbar.set_label("$eE_{x}/mc\omega_\mathrm{L}$")
        else:
            cbar.set_label("$eE_{z}/mc\omega_\mathrm{L}$")
        cbar.ax.minorticks_on()
        cbar.outline.set_linewidth(0.5)
        if E_log:
            offset = (end_x - start_x) * 0.2
            ax.plot(y, E_y.T * AMP + offset, color="grey")
            ax.axhline(offset, color="grey", ls="-.")
    # Axis and labels
    ax.set_title(r"$t =$ " + str(round(time * 1e15, 0)) + " fs")
    ax.set_ylabel(r"$x~(\mathrm{\mu m})$")
    ax.set_xlabel(r"$y~(\mathrm{\mu m})$")
    ax.minorticks_on()

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    plt.tight_layout()

    plt.show()
    # Save figures
    if save:
        run_dir = get_file(filename)
        # save in png format
        fig.savefig(
            rf"{run_dir}/" + str(species) + "_Den_PIConGPU_" + str(iteration) + ".png",
            format="png",
            dpi=600,
            transparent=False,
            bbox_inches="tight",
        )
        # save in pdf format
        fig.savefig(
            rf"{run_dir}/" + str(species) + "_Den_PIConGPU_" + str(iteration) + ".svg",
            format="svg",
            dpi=600,
            transparent=False,
            bbox_inches="tight",
        )
        # save in eps format
    # fig.savefig(rf"{run_dir}/"+str(species)+"_Den_PIConGPU_"+str(iteration)
    # *0.001    +".ps",format='ps',dpi=600, transparent=False, bbox_inches='tight')


def get_h5path(filename):
    """
    Parameter
    ---------
    filename : string
        The name of simulation output
        e.g. lwfa013

    Returns
    -------
    The path of the HDF5 files
    e.g /media/{user}/WORKDIR2/lwfa013/
        simOutput/h5
    """
    run_dir = get_file(filename)
    h5_dir = rf"{run_dir}/simOutput/h5"
    return h5_dir


def get_file(filename):
    """
    Parameter
    ---------
    filename : string
        The name of simulation output
        e.g. lwfa013

    Returns
    -------
    The path of the simulation output
    e.g /media/{user}/WORKDIR2/lwfa013
    """
    run_dir = rf"{path}/{filename}"
    return run_dir


def get_time(filename, iteration):
    """
    Parameter
    ---------
    filename : string
        The name of simulation output
        e.g. lwfa013

    iteration : int
        The iteration at which to obtain the data

    Returns
    -------
    The time of the itaration in second
    e.g 2.10936e-12
    """
    h5_dir = get_h5path(filename)
    h5f = h5py.File(rf"/{h5_dir}/simData_" + str(iteration) + ".h5", "r")
    T = h5f[f"data/{iteration}/"]
    time = T.attrs["time"] * T.attrs["timeUnitSI"]
    return time

def get_username():
    return pwd.getpwuid(os.getuid())[0]


user = get_username()


path = rf"/home/{user}/Development/lwfa_plotter"

pi = 3.1415926535897932384626
v0 = 2.99792458e8  # m/s^2
wavelength = 0.8e-6
epsilon0 = 8.8541878176203899e-12  # F/m
q0 = 1.602176565e-19  # C
m0 = 9.10938291e-31  # kg
frequency = v0 * 2 * pi / wavelength
denunit = frequency ** 2 * epsilon0 * m0 / q0 ** 2
exunit = m0 * v0 * frequency / q0

if __name__ == "__main__":
    get_snapshot(
        "betatron0007",
        iteration=50000,
        species="e",
        simDim="3D",
        movingWin=True,
        cmap1="viridis",
        field=True,
        E_log=True,
        cmap2="Reds_alpha",
        AMP=25,
        save=False,
        slice="z",
    )
