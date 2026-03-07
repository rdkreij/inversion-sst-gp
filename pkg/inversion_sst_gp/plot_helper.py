import re
from operator import sub

import cmocean
import colorcet
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import Affine2D


def imshow(ax, X, Y, Z, *args, **kwargs):
    # modified imshow
    if X.ndim == 1:
        dx = (X[1] - X[0]) / 2
        dy = (Y[1] - Y[0]) / 2
        extent = [X[0] - dx, X[-1] + dx, Y[0] - dy, Y[-1] + dy]
    elif X.ndim == 2:
        dx = (X[0, 1] - X[0, 0]) / 2
        dy = (Y[1, 0] - Y[0, 0]) / 2
        extent = [X[0, 0] - dx, X[0, -1] + dx, Y[0, 0] - dy, Y[-1, 0] + dy]
    else:
        raise ValueError("X must be 1D or 2D")
    return ax.imshow(np.flip(Z, axis=0), extent=extent, *args, **kwargs)


def mask(ax, X, Y, mask, color, *args, **kwargs):
    maskn = np.zeros(mask.shape)
    maskn[np.logical_not(mask)] = np.nan
    cmap = mcolors.ListedColormap([color])
    return imshow(ax, X, Y, maskn, cmap=cmap, *args, **kwargs)


def quiver(
    ax,
    x,
    y,
    u,
    v,
    scale=1.0,
    unit=1.0,
    label="",
    xlims=None,
    ylims=None,
    nx=None,
    ny=None,
    an=True,
    **kwargs,
):
    # quiver with annotation
    if nx is None:
        ix = 1
    else:
        ix = int(round(x.shape[1] / nx))

    if ny is None:
        iy = 1
    else:
        iy = int(round(y.shape[0] / ny))

    vec = ax.quiver(
        x[::iy, ::ix],
        y[::iy, ::ix],
        u[::iy, ::ix],
        v[::iy, ::ix],
        units="width",
        scale=scale,
        **kwargs,
    )

    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()

    if an:
        dx = xlims[1] - xlims[0]
        dy = ylims[1] - ylims[0]

        ax.quiver(
            xlims[1] - dx * unit / scale,
            ylims[1] + dy * 0.03,
            unit,
            0,
            units="width",
            scale=scale,
            clip_on=False,
        )
        ax.annotate(
            str(unit) + " " + label + " ",
            xy=(1 - unit / scale, 1.03),
            xycoords="axes fraction",
            va="center",
            ha="right",
        )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    return vec


def draw_confidence_ellipse_single(
    ax, x, y, cov, n_std=1, scale=1, xlims=None, ylims=None, **kwargs
):
    # plot a single confidence ellipse
    ax.set_aspect(1)
    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    dx = xlims[1] - xlims[0]

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    obj = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs
    )  # draw sample ellipse
    scale_x = np.sqrt(cov[0, 0]) * n_std  # standard deviation of x
    scale_y = np.sqrt(cov[1, 1]) * n_std  # standard deviation of y
    mean_x = x
    mean_y = y

    # transform object
    transf = (
        Affine2D()
        .rotate_deg(45)
        .scale(scale_x / scale * dx, scale_y / scale * dx / get_aspect(ax))
        .translate(mean_x, mean_y)
    )
    obj.set_transform(transf + ax.transData)
    ax.add_patch(obj)
    return ax


def anno_confidence_ellipse(
    ax, scale=1, unit=1, label="", xlims=None, ylims=None, **kwargs
):
    # add annotation to plot of confidence ellipse(s)
    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    dx = xlims[1] - xlims[0]

    arrowprops = dict(
        arrowstyle="|-|,widthA=0.35,widthB=0.35",
        linewidth=1,
        shrinkA=0,
        shrinkB=0,
        **kwargs,
    )
    ax.annotate(
        "",
        xy=(1, 1.03),
        xycoords="axes fraction",
        xytext=(1 - unit / dx / scale * dx, 1.03),
        textcoords="axes fraction",
        arrowprops=arrowprops,
    )
    ax.annotate(
        str(unit) + " " + label + " ",
        xy=(1 - unit / dx / scale * dx, 1.03),
        xycoords="axes fraction",
        va="center",
        ha="right",
    )
    return ax


def draw_confidence_ellipse(
    ax,
    x,
    y,
    c_cov,
    nx=None,
    ny=None,
    n_std=1,
    scale=1,
    unit=1,
    label="",
    xlims=None,
    ylims=None,
    an=True,
    **kwargs,
):
    # plot confidence ellipses
    Ny, Nx = x.shape
    if nx is None:
        ix = 1
    else:
        ix = int(round(Nx / nx))

    if ny is None:
        iy = 1
    else:
        iy = int(round(Ny / ny))

    for i in np.arange(0, Nx, ix):
        for j in np.arange(0, Ny, iy):
            draw_confidence_ellipse_single(
                ax,
                x[j, i],
                y[j, i],
                c_cov[j, i, :, :],
                n_std=n_std,
                scale=scale,
                xlims=xlims,
                ylims=ylims,
                **kwargs,
            )
    if an:
        ax = anno_confidence_ellipse(
            ax, scale=scale, unit=unit, label=label, xlims=xlims, ylims=ylims, color="k"
        )
    return ax


def get_aspect(ax):
    # get the aspect ratio of the figure
    figW, figH = ax.get_figure().get_size_inches()  # total figure size
    _, _, w, h = ax.get_position().bounds  # axis size on figure
    disp_ratio = (figH * h) / (figW * w)  # ratio of display units
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())  # ratio of data units
    return disp_ratio / data_ratio


def ticks_from_spacing(lims, tickspacing):
    # generate ticks based on a specified tick spacing and the data range

    # calculate the order of magnitude of the tick spacing
    order = int(np.floor(np.log10(abs(tickspacing))))

    # calculate the lower and upper limits for tick generation
    lower_limit = int(np.floor(lims[0] * 10 ** (-order - 1))) * 10 ** (order + 1)
    upper_limit = int(np.ceil(lims[1] * 10 ** (-order - 1))) * 10 ** (order + 1)

    # generate all possible ticks
    ticks_all = np.arange(lower_limit, upper_limit + tickspacing, tickspacing)

    # filter out ticks that fall outside the data range
    ticks = ticks_all[(ticks_all >= lims[0]) & (ticks_all <= lims[1])]

    return ticks, order


def ticks_in_deg(ax, xtickspacing=None, ytickspacing=None):
    # change tick labels into geographic coordinates.
    if xtickspacing is not None:
        xlims = ax.get_xlim()
        xticks, xorder = ticks_from_spacing(xlims, xtickspacing)
        Nx = len(xticks)
        xticklabels = []

        for i in range(Nx):
            xtick = xticks[i]
            if xtick >= 0:
                if xorder >= 0:
                    xticklabels.append(("{:.0f}").format(xtick) + "$^\circ$E")
                else:
                    xticklabels.append(
                        ("{:." + str(-xorder) + "f}").format(xtick) + "$^\circ$E"
                    )
            else:
                if xorder > -0:
                    xticklabels.append(("{:.0f}").format(-xtick) + "$^\circ$W")
                else:
                    xticklabels.append(
                        ("{:." + str(-xorder) + "f}").format(-xtick) + "$^\circ$W"
                    )

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    if ytickspacing is not None:
        ylims = ax.get_ylim()
        yticks, yorder = ticks_from_spacing(ylims, ytickspacing)
        Ny = len(yticks)
        yticklabels = []

        for i in range(Ny):
            ytick = yticks[i]
            if ytick >= 0:
                if yorder >= 0:
                    yticklabels.append(("{:.0f}").format(ytick) + "$^\circ$N")
                else:
                    yticklabels.append(
                        ("{:." + str(-yorder) + "f}").format(ytick) + "$^\circ$N"
                    )
            else:
                if yorder > -0:
                    yticklabels.append(("{:.0f}").format(-ytick) + "$^\circ$S")
                else:
                    yticklabels.append(
                        ("{:." + str(-yorder) + "f}").format(-ytick) + "$^\circ$S"
                    )

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)


def add_colorbar_map(
    fig, cax, lims, cmap, side, label=None, format_label=False, **kwargs
):
    # add colormap to axis

    # value limits
    norm = Normalize(vmin=lims[0], vmax=lims[1])
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # placing and tick locations
    if (side == "left") or (side == "right"):
        plt.colorbar(sm, cax=cax, orientation="vertical", **kwargs)
        cax.yaxis.set_ticks_position(side)
        cax.yaxis.set_label_position(side)
        if side == "left":
            cax.yaxis.set_offset_position("right")
        elif side == "right":
            cax.yaxis.set_offset_position("left")
        if label is not None:
            if format_label:
                add_ax_ylabel_format(fig, cax, label)
            else:
                cax.set_ylabel(label)
    elif (side == "bottom") or (side == "top"):
        plt.colorbar(sm, cax=cax, orientation="horizontal", **kwargs)
        cax.xaxis.set_ticks_position(side)
        cax.xaxis.set_label_position(side)
        if label is not None:
            if format_label:
                add_ax_xlabel_format(fig, cax, label)
            else:
                cax.set_xlabel(label)
    pass


def add_colorbar(
    fig, ax, side, lims, cmap, cbr, cbd, cbw, label=None, format_label=False, **kwargs
):
    # add colorbar

    pos = ax.get_position()  # get the position of the axes
    x0, y0, pw, ph = (
        pos.x0,
        pos.y0,
        pos.width,
        pos.height,
    )  # extract the location, width, and height
    if side == "top":
        cax = fig.add_axes([x0 + pw * (1 - cbr) / 2, y0 + ph + cbd, cbr * pw, cbw])
    elif side == "bottom":
        cax = fig.add_axes([x0 + pw * (1 - cbr) / 2, y0 - cbd - cbw, cbr * pw, cbw])
    elif side == "left":
        cax = fig.add_axes([x0 - cbd - cbw, y0 + ph * (1 - cbr) / 2, cbw, cbr * ph])
    elif side == "right":
        cax = fig.add_axes([x0 + pw + cbd, y0 + ph * (1 - cbr) / 2, cbw, cbr * ph])
    else:
        raise ValueError("side must be 'top', 'bottom', 'left', or 'right'")
    add_colorbar_map(
        fig, cax, lims, cmap, side, label=label, format_label=format_label, **kwargs
    )
    pass


def add_ax_ylabel_format(fig, ax, label):
    fig.canvas.draw()
    offset_text = ax.yaxis.get_offset_text().get_text()

    if offset_text:
        ax.yaxis.offsetText.set_visible(False)
        ax.set_ylabel(f"{label} {offset_text}")
    else:
        ax.set_ylabel(label)


def add_ax_xlabel_format(fig, ax, label):
    fig.canvas.draw()
    offset_text = ax.xaxis.get_offset_text().get_text()

    if offset_text:
        ax.xaxis.offsetText.set_visible(False)
        ax.set_xlabel(f"{label} {offset_text}")
    else:
        ax.set_xlabel(label)


# def annotate_corner(ax, text, x=.026, y=.974):
#     # annotate corner of plot


#     ax.annotate(text,
#                 xy=(x,y),
#                 va='top',ha='left',
#                 xycoords='axes fraction',
#                 bbox=dict(facecolor='white',edgecolor='black',pad=3.5,linewidth=1),
#                 clip_box=ax.bbox,
#                 zorder=10)
#     pass
def annotate_corner(ax, text, fontsize=12, fig=None, box=True, buffer=0):
    if fig is None:
        fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_height_inches = bbox.height
    axis_height_points = axis_height_inches * 72  # Convert inches to points
    axis_width_inches = bbox.width
    axis_width_points = axis_width_inches * 72  # Convert inches to points

    aspect = 0.6
    me = 2 * fontsize / 12 + buffer
    ms = 2 * fontsize / 12 + buffer
    mn = 1 * fontsize / 12 + buffer
    mw = 3 * fontsize / 12 + buffer

    box_h = (fontsize + mn + ms) / axis_height_points
    box_w = (fontsize * aspect + me + mw) / axis_width_points

    if box:
        # Add a square in the top right corner of the axis
        corner_square = Rectangle(
            (0, 1 - box_h),  # (x, y) position of bottom left corner in axis fraction
            box_w,  # width of the square in axis fraction
            box_h,  # height of the square in axis fraction
            transform=ax.transAxes,  # Use axis coordinates for positioning
            facecolor="w",  # Color of the square
            edgecolor="k",  # Transparency
            zorder=99,
        )
        ax.add_patch(corner_square)
    ax.text(
        mw / axis_width_points,
        1 - (mn + fontsize) / axis_height_points,
        text,
        va="bottom",
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
        zorder=100,
    )


def visualize_data(
    LON,
    LAT,
    T,
    dTdt,
    dTds1,
    dTds2,
    u=None,
    v=None,
    lonlims=None,
    latlims=None,
    plimT=None,
    plimTgrad=None,
    plimdTdt=None,
    plimspeed=None,
    pscale=4,
    nx=17,
    ny=17,
    plt_show=False,
    return_fig=False,
    label_obs=True,
):
    bool_velocity = (u is not None) and (v is not None)

    mn = 0.3
    mw = 0.2
    me = 0.03
    ms = 0.18
    md = 0

    Nx = 3 + bool_velocity
    Ny = 1

    aspect = 1

    lx = me + mw + md * (Nx - 1) + Nx * aspect
    ly = mn + ms + Ny

    ph = aspect / ly
    pw = 1 / lx

    cbd = 0.01 * lx / ly
    cbr = 0.85
    cbw = 0.015 * lx / ly

    px_range = (mw + np.arange(Nx) * (1 + md)) / lx
    py_range = (ly - mn - 1 - np.arange(Ny)) / ly

    fw = Nx * 2.5
    fh = fw * ly / lx

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes(
                (px_range[i], py_range[jj], pw, ph)
            )  # [left, bottom, width, height]
            if lonlims is not None:
                ax[jj, i].set_xlim(lonlims)
            if latlims is not None:
                ax[jj, i].set_ylim(latlims)

    cmapT = cmocean.cm.thermal
    cmapdTdt = cmocean.cm.haline
    cmapTgrad = cmocean.cm.amp

    if plimT is None:
        plimT = [np.nanmin(T), np.nanmax(T)]

    if plimTgrad is None:
        listvar = [np.sqrt(dTds1**2 + dTds2**2).flatten()]
        plimvalTgrad = np.hstack(listvar)
        plimTgrad = [0, np.nanmax(plimvalTgrad)]

    if plimdTdt is None:
        listvar = [dTdt.flatten()]
        plimvaldTdt = np.hstack(listvar)
        plimdTdt = [np.nanmin(plimvaldTdt), np.nanmax(plimvaldTdt)]

    if label_obs:
        add_colorbar(
            fig,
            ax[0, 0],
            "top",
            plimT,
            cmapT,
            cbr,
            cbd,
            cbw,
            label=r"$\mathbf{T}^o$ ($^\circ$C)",
        )
        add_colorbar(
            fig,
            ax[0, 1],
            "top",
            plimTgrad,
            cmapTgrad,
            cbr,
            cbd,
            cbw,
            label=r"$\nabla \mathbf{T}^o$ (K$\,$m$^{-1}$)",
        )
        add_colorbar(
            fig,
            ax[0, 2],
            "top",
            plimdTdt,
            cmapdTdt,
            cbr,
            cbd,
            cbw,
            label=r"$d\mathbf{T}^o/dt$ (K$\,$s$^{-1}$)",
        )
    else:
        add_colorbar(
            fig,
            ax[0, 0],
            "top",
            plimT,
            cmapT,
            cbr,
            cbd,
            cbw,
            label=r"$\mathbf{T}$ ($^\circ$C)",
        )
        add_colorbar(
            fig,
            ax[0, 1],
            "top",
            plimTgrad,
            cmapTgrad,
            cbr,
            cbd,
            cbw,
            label=r"$\nabla \mathbf{T}$ (K$\,$m$^{-1}$)",
        )
        add_colorbar(
            fig,
            ax[0, 2],
            "top",
            plimdTdt,
            cmapdTdt,
            cbr,
            cbd,
            cbw,
            label=r"$d\mathbf{T}/dt$ (K$\,$s$^{-1}$)",
        )

    imshow(ax[0, 0], LON, LAT, T, cmap=cmapT, vmin=plimT[0], vmax=plimT[1])
    imshow(
        ax[0, 1],
        LON,
        LAT,
        np.sqrt(dTds1**2 + dTds2**2),
        cmap=cmapTgrad,
        vmin=0,
        vmax=plimTgrad[1],
    )
    imshow(ax[0, 2], LON, LAT, dTdt, cmap=cmapdTdt, vmin=plimdTdt[0], vmax=plimdTdt[1])

    mask(ax[0, 2], LON, LAT, np.isnan(dTdt), "white")
    mask(ax[0, 0], LON, LAT, np.isnan(T), "white")

    if bool_velocity:
        cmapspeed = cmocean.tools.lighten(cmocean.cm.speed, 0.6)
        if plimspeed is None:
            listvar = [np.sqrt(u**2 + v**2).flatten()]
            plimvalspeed = np.hstack(listvar)
            plimspeed = [0, np.nanmax(plimvalspeed)]
        add_colorbar(
            fig,
            ax[0, 3],
            "top",
            plimspeed,
            cmapspeed,
            cbr,
            cbd,
            cbw,
            label=r"$(\mathbf{u}',\mathbf{v}')'$ (m$\,$s$^{-1}$)",
        )

        imshow(
            ax[0, 3],
            LON,
            LAT,
            np.sqrt(u**2 + v**2),
            cmap=cmapspeed,
            vmin=plimspeed[0],
            vmax=plimspeed[1],
        )
        quiver(
            ax[0, 3], LON, LAT, u, v, nx=nx, ny=ny, scale=pscale, an=False, color="k"
        )

        ldx = 0.01
        ldy = -0.03 * lx / ly
        lh = 0.015 * lx / ly
        lw = 0.005

        mw = px_range[3]
        ms = py_range[0]

        ax0.text(
            mw + ldx, ms + lh + ldy, "Velocity", ha="left", va="center", fontsize=10
        )
        ax0.annotate(
            "",
            xy=(mw + ldx, ms + ldy),
            xytext=(mw + pw / pscale + ldx, ms + ldy),
            arrowprops=dict(
                arrowstyle="<|-,widthA=0.35,widthB=0.35",
                linewidth=1,
                facecolor="k",
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax0.text(
            mw + pw / pscale + lw + ldx,
            ms + ldy,
            "1 m$\,$s$^{-1}$",
            ha="left",
            va="center",
            fontsize=10,
        )

    i = 0
    for axi in ax.flatten():
        if axi.has_data():
            ticks_in_deg(axi, 1, 1)
            axi.xaxis.set_ticks_position("both")
            axi.yaxis.set_ticks_position("both")
            axi.tick_params(axis="both", direction="in", length=5)
            axi.set_facecolor("white")
            i += 1

    jticks = 0
    iticks = 0
    for i in range(Nx):
        for j in range(Ny):
            if (i != iticks) or (j != jticks):
                ax[j, i].set_xticklabels([])
                ax[j, i].set_yticklabels([])

    ax0.axis("off")

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)
        return


def plot_predictions_osse(
    LON,
    LAT,
    To,
    dTds1o,
    dTds2o,
    dTdto,
    muSstar,
    Kxstar_vel,
    stdSstar,
    muustar,
    muvstar,
    lonlims,
    latlims,
    u=None,
    v=None,
    S=None,
    LONr=None,
    LATr=None,
    ur=None,
    vr=None,
    ugos=None,
    vgos=None,
    Sgos=None,
    params=None,
    datet=None,
    plimT=None,
    plimdTdt=None,
    plimTgrad=None,
    plimstdS=None,
    plimspeed=None,
    pscale=4,
    nx=17,
    ny=17,
    pscale_cred=None,
    plt_show=False,
    return_fig=False,
    nxr=None,
    nyr=None,
):
    # plot overview of satellite products and predictions

    if pscale_cred is None:
        pscale_cred = pscale

    bool_original_data = (u is not None) & (v is not None) & (S is not None)
    bool_reference_velocity = (
        (LONr is not None) & (LATr is not None) & (ur is not None) & (vr is not None)
    )
    bool_GOS = (ugos is not None) & (vgos is not None) & (Sgos is not None)

    mn = 0.4
    mw = 0.15
    me = 0.25
    if bool_reference_velocity:
        ms = 0.5
    else:
        ms = 0.4
    md = 0

    td = 0.005

    Nx = 4
    Ny = 3 + bool_GOS - bool_reference_velocity

    aspect = 1

    lx = me + mw + Nx * aspect
    ly = mn + ms + md * (Ny - 1) + Ny

    ph = aspect / ly
    pw = 1 / lx

    cbd = 0.01 * lx / ly
    cbr = 0.85
    cbw = 0.02 * lx / ly

    px_range = (mw + np.arange(Nx)) / lx
    py_range = (ly - mn - 1 - np.arange(Ny) * (1 + md)) / ly

    fw = 8
    fh = fw * ly / lx

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes(
                [px_range[i], py_range[jj], pw, ph]
            )  # [left, bottom, width, height]
            ax[jj, i].set_xlim(lonlims)
            ax[jj, i].set_ylim(latlims)

    cmapstdS = colorcet.cm.gouldian
    cmapTgrad = cmocean.cm.amp
    cmapT = cmocean.cm.thermal
    cmapdTdt = cmocean.cm.haline
    cmapspeed = cmocean.tools.lighten(cmocean.cm.speed, 0.6)

    def set_lims(plim, datalim):
        if plim is None:
            return datalim, "neither"
        else:
            extend_min = False
            extend_max = False
            if plim[0] is None:
                lim = [datalim[0]]
            else:
                lim = [plim[0]]
                if plim[0] > datalim[0]:
                    extend_min = True

            if plim[1] is None:
                lim.append(datalim[1])
            else:
                lim.append(plim[1])
                if plim[1] < datalim[1]:
                    extend_max = True

            if extend_min and extend_max:
                extend = "both"
            elif extend_min:
                extend = "min"
            elif extend_max:
                extend = "max"
            else:
                extend = "neither"
            return lim, extend

    datalimT = [np.nanmin(To), np.nanmax(To)]
    plimT, extendT = set_lims(plimT, datalimT)

    listvar = [muSstar.flatten(), dTdto.flatten()]
    if bool_GOS:
        listvar += [Sgos.flatten()]
    if bool_original_data:
        listvar += [S.flatten()]
    datalimvaldTdt = np.hstack(listvar)
    datalimdTdt = [np.nanmin(datalimvaldTdt), np.nanmax(datalimvaldTdt)]
    plimdTdt, extenddTdt = set_lims(plimdTdt, datalimdTdt)

    datalimvalTgrad = np.sqrt(dTds1o**2 + dTds2o**2)
    datalimTgrad = [0, np.nanmax(datalimvalTgrad)]
    plimTgrad, extendTgrad = set_lims(plimTgrad, datalimTgrad)

    datalimvalstdS = np.concatenate([stdSstar])
    datalimstdS = [0, np.nanmax(np.abs(datalimvalstdS))]
    plimstdS, extendstdS = set_lims(plimstdS, datalimstdS)

    listvar = [np.sqrt(muustar**2 + muvstar**2).flatten()]
    if bool_GOS:
        listvar += [np.sqrt(ugos**2 + vgos**2).flatten()]
    if bool_original_data:
        listvar += [np.sqrt(u**2 + v**2).flatten()]
    if bool_reference_velocity:
        listvar += [np.sqrt(ur**2 + vr**2).flatten()]
    datalimvalspeed = np.hstack(listvar)
    datalimspeed = [0, np.nanmax(datalimvalspeed)]
    plimspeed, extendspeed = set_lims(plimspeed, datalimspeed)

    add_colorbar(
        fig,
        ax[0, 1],
        "top",
        plimTgrad,
        cmapTgrad,
        cbr,
        cbd,
        cbw,
        label="(K$\,$m$^{-1}$)",
        extend=extendTgrad,
    )
    if bool_reference_velocity:
        add_colorbar(
            fig,
            ax[0, 3],
            "top",
            plimT,
            cmapT,
            cbr,
            cbd,
            cbw,
            label="($^\circ$C)",
            extend=extendT,
        )
        add_colorbar(
            fig,
            ax[1, 3],
            "bottom",
            plimstdS,
            cmapstdS,
            cbr,
            cbd * 3,
            cbw,
            label="(K$\,$s$^{-1}$)",
            extend=extendstdS,
        )
        add_colorbar(
            fig,
            ax[0, 0],
            "top",
            plimspeed,
            cmapspeed,
            cbr,
            cbd,
            cbw,
            label="(m$\,$s$^{-1}$)",
            extend=extendspeed,
        )
    else:
        add_colorbar(
            fig,
            ax[0, 0],
            "top",
            plimT,
            cmapT,
            cbr,
            cbd,
            cbw,
            label="($^\circ$C)",
            extend=extendT,
        )
        add_colorbar(
            fig,
            ax[2, 3],
            "top",
            plimstdS,
            cmapstdS,
            cbr,
            cbd,
            cbw,
            label="(K$\,$s$^{-1}$)",
            extend=extendstdS,
        )
        add_colorbar(
            fig,
            ax[-1, 0],
            "bottom",
            plimspeed,
            cmapspeed,
            cbr,
            cbd,
            cbw,
            label="(m$\,$s$^{-1}$)",
            extend=extendspeed,
        )
    add_colorbar(
        fig,
        ax[0, 2],
        "top",
        plimdTdt,
        cmapdTdt,
        cbr,
        cbd,
        cbw,
        label="(K$\,$s$^{-1}$)",
        extend=extenddTdt,
    )

    if bool_reference_velocity:
        ii = 3
    else:
        ii = 0
    imshow(ax[0, ii], LON, LAT, To, cmap=cmapT, vmin=plimT[0], vmax=plimT[1])
    imshow(
        ax[0, 1],
        LON,
        LAT,
        np.sqrt(dTds1o**2 + dTds2o**2),
        cmap=cmapTgrad,
        vmin=plimTgrad[0],
        vmax=plimTgrad[1],
    )
    imshow(ax[0, 2], LON, LAT, dTdto, cmap=cmapdTdt, vmin=plimdTdt[0], vmax=plimdTdt[1])

    if bool_reference_velocity:
        imshow(
            ax[0, 0],
            LONr,
            LATr,
            np.sqrt(ur**2 + vr**2),
            cmap=cmapspeed,
            vmin=plimspeed[0],
            vmax=plimspeed[1],
        )
        quiver(
            ax[0, 0],
            LONr,
            LATr,
            ur,
            vr,
            scale=pscale,
            an=False,
            color="b",
            nx=nxr,
            ny=nyr,
        )
    elif bool_original_data:
        ax[0, 3].axis("off")
        ax[1, 1].axis("off")
        imshow(
            ax[1, 0],
            LON,
            LAT,
            np.sqrt(u**2 + v**2),
            cmap=cmapspeed,
            vmin=plimspeed[0],
            vmax=plimspeed[1],
        )
        quiver(
            ax[1, 0], LON, LAT, u, v, nx=nx, ny=ny, scale=pscale, an=False, color="r"
        )
        imshow(ax[1, 2], LON, LAT, S, cmap=cmapdTdt, vmin=plimdTdt[0], vmax=plimdTdt[1])
        ax[1, 3].axis("off")

    if bool_reference_velocity:
        jj = 1
    else:
        jj = 2
    imshow(
        ax[jj, 0],
        LON,
        LAT,
        np.sqrt(muustar**2 + muvstar**2),
        cmap=cmapspeed,
        vmin=plimspeed[0],
        vmax=plimspeed[1],
    )
    if bool_original_data:
        quiver(
            ax[jj, 0], LON, LAT, u, v, nx=nx, ny=ny, scale=pscale, an=False, color="r"
        )
    quiver(ax[jj, 0], LON, LAT, muustar, muvstar, nx=nx, ny=ny, scale=pscale, an=False)
    draw_confidence_ellipse(
        ax[jj, 1],
        LON,
        LAT,
        Kxstar_vel,
        nx=nx,
        ny=ny,
        scale=pscale_cred,
        color="grey",
        an=False,
    )
    imshow(
        ax[jj, 2], LON, LAT, muSstar, cmap=cmapdTdt, vmin=plimdTdt[0], vmax=plimdTdt[1]
    )
    imshow(
        ax[jj, 3], LON, LAT, stdSstar, cmap=cmapstdS, vmin=plimstdS[0], vmax=plimstdS[1]
    )

    if bool_original_data:
        ax0.text(
            px_range[0] - td,
            py_range[0],
            "Observing system simulation experiment data",
            ha="right",
            va="center",
            fontsize=12,
            rotation=90,
        )
    if bool_reference_velocity:
        ax0.text(
            px_range[0] - td,
            py_range[0] + ph / 2,
            "Satellite products",
            ha="right",
            va="center",
            fontsize=12,
            rotation=90,
        )
    ax0.text(
        px_range[0] - td,
        py_range[jj] + ph / 2,
        "GP prediction",
        ha="right",
        va="center",
        fontsize=12,
        rotation=90,
    )

    if bool_GOS:
        imshow(
            ax[3, 0],
            LON,
            LAT,
            np.sqrt(ugos**2 + vgos**2),
            cmap=cmapspeed,
            vmin=plimspeed[0],
            vmax=plimspeed[1],
        )
        quiver(
            ax[3, 0], LON, LAT, u, v, nx=nx, ny=ny, scale=pscale, an=False, color="r"
        )
        if bool_original_data:
            quiver(
                ax[3, 0],
                LON,
                LAT,
                u,
                v,
                nx=nx,
                ny=ny,
                scale=pscale,
                an=False,
                color="r",
            )
        quiver(ax[3, 0], LON, LAT, ugos, vgos, nx=nx, ny=ny, scale=pscale, an=False)
        imshow(
            ax[3, 2], LON, LAT, Sgos, cmap=cmapdTdt, vmin=plimdTdt[0], vmax=plimdTdt[1]
        )
        ax[3, 1].axis("off")
        ax[3, 3].axis("off")

        ax0.text(
            px_range[0] - td,
            py_range[3] + ph / 2,
            "GOS",
            ha="right",
            va="center",
            fontsize=12,
            rotation=90,
        )

    mask(ax[0, 1], LON, LAT, np.isnan(np.sqrt(dTds1o**2 + dTds2o**2)), "white")
    mask(ax[0, 2], LON, LAT, np.isnan(dTdto), "white")
    mask(ax[0, ii], LON, LAT, np.isnan(To), "white")

    i = 0
    for axi in ax.flatten():
        if axi.has_data():
            ticks_in_deg(axi, 1, 1)
            axi.xaxis.set_ticks_position("both")
            axi.yaxis.set_ticks_position("both")
            axi.tick_params(
                axis="both", direction="in", length=5, labelright=True, labelleft=False
            )

            annotate_corner(axi, chr(97 + i))
            i += 1

    jticks = Ny - 1
    if bool_GOS:
        iticks = 2
    else:
        iticks = 3
    for i in range(Nx):
        for j in range(Ny):
            if (i != iticks) or (j != jticks):
                ax[j, i].set_xticklabels([])
                ax[j, i].set_yticklabels([])

    if bool_reference_velocity:
        mw = px_range[1]
        ms = py_range[1]

        ldx = 0.02
        ldy = -0.05 * lx / ly
        lh = 0.02 * lx / ly
        lw = 0.005
        ax0.text(
            mw + ldx,
            ms + lh + ldy,
            "1 SD Credible Ellipse",
            ha="left",
            va="center",
            fontsize=10,
        )
        ax0.annotate(
            "",
            xy=(mw + ldx, ms + ldy),
            xytext=(mw + pw / (pscale_cred) + ldx, ms + ldy),
            arrowprops=dict(
                arrowstyle="|-|,widthA=0.35,widthB=0.35",
                linewidth=1,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax0.text(
            mw + pw / pscale_cred + lw + ldx,
            ms + ldy,
            "1 m$\,$s$^{-1}$",
            ha="left",
            va="center",
            fontsize=10,
        )

        mw = px_range[0]
        ms = py_range[1]

        ax0.text(
            mw + ldx, ms + lh + ldy, "Velocity", ha="left", va="center", fontsize=10
        )
        ax0.annotate(
            "",
            xy=(mw + ldx, ms + ldy),
            xytext=(mw + pw / pscale + ldx, ms + ldy),
            arrowprops=dict(
                arrowstyle="<|-,widthA=0.35,widthB=0.35",
                linewidth=1,
                facecolor="k",
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax0.text(
            mw + pw / pscale + lw + ldx,
            ms + ldy,
            "1 m$\,$s$^{-1}$",
            ha="left",
            va="center",
            fontsize=10,
        )
    else:
        mw = px_range[1]
        ms = py_range[1]

        ldx = 0.02
        ldy = 0.02 * lx / ly
        lh = 0.02 * lx / ly
        lw = 0.005
        ax0.text(
            mw + ldx,
            ms + lh + ldy,
            "1 SD Credible Ellipse",
            ha="left",
            va="center",
            fontsize=10,
        )
        ax0.annotate(
            "",
            xy=(mw + ldx, ms + ldy),
            xytext=(mw + pw / (pscale) + ldx, ms + ldy),
            arrowprops=dict(
                arrowstyle="|-|,widthA=0.35,widthB=0.35",
                linewidth=1,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax0.text(
            mw + pw / (pscale) + lw + ldx,
            ms + ldy,
            "1 m$\,$s$^{-1}$",
            ha="left",
            va="center",
            fontsize=10,
        )

        ax0.text(
            mw + ldx, ms + 4 * lh + ldy, "Velocity", ha="left", va="center", fontsize=10
        )
        ax0.annotate(
            "",
            xy=(mw + ldx, ms + 3 * lh + ldy),
            xytext=(mw + pw / pscale + ldx, ms + 3 * lh + ldy),
            arrowprops=dict(
                arrowstyle="<|-,widthA=0.35,widthB=0.35",
                linewidth=1,
                facecolor="k",
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax0.text(
            mw + pw / pscale + lw + ldx,
            ms + 3 * lh + ldy,
            "1 m$\,$s$^{-1}$",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax0.axis("off")
    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)
        return


def lon_transect(
    ax,
    lon,
    z_mu,
    z_std,
    lonlims,
    xtickspacing,
    z_true=None,
    title="",
    ylabel="",
    xlabel="",
    fig=None,
):
    # assign figure
    if fig is None:
        fig = plt.gcf()

    # set axis limits
    ax.set(xlim=lonlims)

    # set tick spacing and labels in degrees
    ticks_in_deg(ax=ax, xtickspacing=xtickspacing)

    ax.plot(lon, z_mu, "k-", linewidth=1, label="$\mu_u^*$")
    ax.fill_between(
        lon,
        z_mu - 2 * z_std,
        z_mu + 2 * z_std,
        facecolor=[0.85, 0.85, 0.85],
        label="1 std(u$^*$)",
    )
    ax.fill_between(
        lon,
        z_mu - 1 * z_std,
        z_mu + 1 * z_std,
        facecolor=[0.7, 0.7, 0.7],
        label="2 std(u$^*$)",
    )

    if z_true is not None:
        ax.plot(lon, z_true, "ro", markersize=2, label="u")

    # set plot title and axis labels
    ax.set_title(title, loc="left")

    # set tick positions and parameters for both axes
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="both", direction="in", length=5)

    # labels
    ax.set(ylabel=ylabel, xlabel=xlabel, title=title)
    pass


def plot_transects(
    transect_fully_observed,
    transect_measurement_error,
    transect_dense_cloud,
    transect_sparse_cloud,
    lonlims,
    ylims,
    plt_show=False,
    return_fig=False,
):
    fig, ax = plt.subplots(
        2, 2, figsize=(10, 4.2), gridspec_kw={"hspace": 0, "wspace": 0}
    )

    for i, dic in enumerate(
        [
            transect_fully_observed,
            transect_measurement_error,
            transect_dense_cloud,
            transect_sparse_cloud,
        ]
    ):
        axi = ax.flatten()[i]
        lon = dic["lon"]
        lon_transect(
            axi,
            lon,
            dic["muvstar"],
            dic["stdvstar"],
            lonlims,
            1,
            z_true=dic["v"],
            fig=None,
        )
        axi.set_ylim(ylims)
        axi.set_xlim(lonlims)

        if i == 2:
            idx_nan = np.where(dic["maskc"])[0]
            clims = [lon[idx_nan[0]], lon[idx_nan[-1]]]
            axi.axvspan(
                clims[0],
                clims[1],
                alpha=1,
                color="k",
                fill=False,
                hatch="//",
                zorder=-5,
                lw=1,
            )

    ax[0, 0].annotate(
        "a) Fully observed",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=12,
    )
    ax[0, 1].annotate(
        "b) Measurement error",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=12,
    )
    ax[1, 0].annotate(
        "c) Dense cloud",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="square,pad=.2", facecolor="w", ec="w"),
    )
    ax[1, 1].annotate(
        "d) Sparse cloud",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=12,
    )

    ax[1, 1].set_xticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 0].set_xticklabels([])

    ax[0, 1].set_yticklabels([])
    ax[1, 1].set_yticklabels([])

    ax[0, 0].set(ylabel="$v_t$ (m$\,$s$^{-1}$)")
    ax[1, 0].set(ylabel="$v_t$ (m$\,$s$^{-1}$)")

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_dynamic_rossby(
    LON,
    LAT,
    Ro,
    lonlims,
    latlims,
    Ro_max=None,
    pdf_max=None,
    u=None,
    v=None,
    nx=17,
    ny=17,
    pscale=4,
    LONr=None,
    LATr=None,
    Ror=None,
    plt_show=False,
    return_fig=False,
):
    if LONr is None:
        bool_ref = False
    else:
        bool_ref = True

    mn = 0.12
    mw = 0.6
    me = 0.1
    ms = 0.18
    md = 0.3
    aspect_r = 1.5

    Nx = 2 + bool_ref
    Ny = 1

    aspect = np.diff(lonlims)[0] / np.diff(latlims)[0]

    lx = me + mw + (Nx - 1) + aspect_r / aspect + md
    ly = mn + ms + Ny / aspect

    ph = 1 / ly / aspect
    pw = [1 / lx] * (Nx - 1) + [aspect_r / aspect / lx]

    cbd = 0.14 * ly / lx
    cbr = 0.85
    cbw = 0.04 * ly / lx

    px_range = [mw / lx] + bool_ref * [(mw + 1) / lx] + [(mw + 1 + bool_ref + md) / lx]
    py_range = np.array([ms / ly])

    if bool_ref:
        fw = 10
    else:
        fw = 8
    fh = fw * ly / lx

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    cmapRo = cmocean.cm.balance
    if Ro_max is None:
        if bool_ref:
            Roval = np.stack([Ro.flatten(), Ror.flatten()])
        else:
            Roval = Ro
        Ro_max = np.max(np.abs(Roval))
    plimRo = [-Ro_max, Ro_max]

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes(
                [px_range[i], py_range[jj], pw[i], ph]
            )  # [left, bottom, width, height]
            if i in range(Nx - 1):
                ax[jj, i].set_xlim(lonlims)
                ax[jj, i].set_ylim(latlims)

    i = 0
    imshow(ax[0, i], LON, LAT, Ro, cmap=cmapRo, vmin=plimRo[0], vmax=plimRo[1])
    add_colorbar(
        fig, ax[0, i], "left", plimRo, cmapRo, cbr, cbd, cbw, label=r"$\textrm{Ro}_t$"
    )

    if u is not None:
        quiver(
            ax[0, i],
            LON,
            LAT,
            u,
            v,
            nx=nx,
            ny=ny,
            scale=pscale,
            an=True,
            l_unit="m$\,$s$^{-1}$",
        )

    if bool_ref:
        i += 1
        ax[0, 0].set(title="GP")
        ax[0, 1].set(title="Altimetry")
        imshow(ax[0, i], LONr, LATr, Ror, cmap=cmapRo, vmin=plimRo[0], vmax=plimRo[1])

    i += 1
    bin_step = 0.04
    range_pos = np.arange(bin_step / 2, Ro_max, bin_step)
    bins = np.hstack([-np.flip(range_pos), range_pos])
    if bool_ref:
        counts, bins = np.histogram(Ro.flatten(), bins=bins, density=True)
        ax[0, i].plot(
            bins[:-1], counts, drawstyle="steps-post", color="black", lw=1, label="GP"
        )
        counts, bins = np.histogram(Ror.flatten(), bins=bins, density=True)
        ax[0, i].plot(
            bins[:-1],
            counts,
            drawstyle="steps-post",
            color="red",
            lw=1,
            label="Altimetry",
        )
        ax[0, i].legend()
    else:
        ax[0, i].hist(Ro.flatten(), bins=bins, density=True, fc="grey", ec="k")
    ax[0, i].set(xlabel=r"$\textrm{Ro}_t$", ylabel="Empirical density")
    if pdf_max is not None:
        ax[0, i].set_ylim([0, pdf_max])
    else:
        _, ymax = ax[0, i].get_ylim()
        ax[0, i].set_ylim([0, ymax])

    i = 0
    for i, axi in enumerate(ax.flatten()):
        if i in range(Nx - 1):
            ticks_in_deg(axi, 1, 1)
        axi.xaxis.set_ticks_position("both")
        axi.yaxis.set_ticks_position("both")
        axi.tick_params(axis="both", direction="in", length=5)

        annotate_corner(axi, chr(97 + i))
        axi.set_facecolor("white")
        i += 1

    if bool_ref:
        ax[0, 1].set_xticklabels([])
        ax[0, 1].set_yticklabels([])

    ax0.axis("off")

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_particle_tracking(
    list_dict_tracks,
    lon,
    lat,
    muustar,
    muvstar,
    Kxstar_vel,
    T,
    lonlimsp,
    latlimsp,
    pscale=1,
    plt_show=False,
    return_fig=False,
):
    LON, LAT = np.meshgrid(lon, lat)

    crop_lon = (lon >= lonlimsp[0]) & (lon <= lonlimsp[1])
    crop_lat = (lat >= latlimsp[0]) & (lat <= latlimsp[1])
    LONc = LON[crop_lat, :][:, crop_lon]
    LATc = LAT[crop_lat, :][:, crop_lon]
    Tc = T[crop_lat, :][:, crop_lon]
    muustarc = muustar[crop_lat, :][:, crop_lon]
    muvstarc = muvstar[crop_lat, :][:, crop_lon]
    Kxstar_velc = Kxstar_vel[crop_lat, :, :, :][:, crop_lon, :, :]

    mn = 0.3
    mw = 0.25
    me = 0.05
    ms = 0.1

    Nx = 3
    Ny = 1

    aspect = np.diff(lonlimsp)[0] / np.diff(latlimsp)[0]

    lx = me + mw + Nx
    ly = mn + ms + Ny / aspect

    ph = 1 / ly / aspect
    pw = 1 / lx

    cbd = 0.01 * lx / ly
    cbr = 0.85
    cbw = 0.02 * lx / ly

    px_range = (mw + np.arange(Nx)) / lx
    py_range = np.array([ms / ly])

    fw = 8
    fh = fw * ly / lx

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes(
                [px_range[i], py_range[jj], pw, ph]
            )  # [left, bottom, width, height]
            ax[jj, i].set_xlim(lonlimsp)
            ax[jj, i].set_ylim(latlimsp)

    cmapT = cmocean.tools.lighten(cmocean.cm.thermal, 0.8)
    plimT = [np.nanmin(Tc), np.nanmax(Tc)]

    add_colorbar(fig, ax[0, 0], "top", plimT, cmapT, cbr, cbd, cbw, label="($^\circ$C)")

    for axi in ax.flatten():
        axi.set_ylim(latlimsp)
        axi.set_xlim(lonlimsp)
        axi.set_aspect("equal")

    pscale_cred = pscale

    imshow(ax[0, 0], LONc, LATc, Tc, cmapT)

    draw_confidence_ellipse(
        ax[0, 2],
        LONc,
        LATc,
        Kxstar_velc,
        scale=pscale_cred,
        an=False,
        edgecolor="grey",
        facecolor="lightgrey",
    )

    for dict_tracks in list_dict_tracks:
        for axi in ax.flatten():
            n_sample = dict_tracks["lons"].shape[0]
            for i in range(n_sample):
                axi.plot(
                    dict_tracks["lons"][i, :],
                    dict_tracks["lats"][i, :],
                    color="k",
                    lw=0.5,
                    alpha=1,
                )
            axi.plot(dict_tracks["lonsm"], dict_tracks["latsm"], "r--", lw=1, alpha=1)
            axi.plot(dict_tracks["lon0"], dict_tracks["lat0"], "ro", ms=6)

    quiver(
        ax[0, 1], LONc, LATc, muustarc, muvstarc, scale=pscale, an=False, color="grey"
    )

    i = 0
    for axi in ax.flatten():
        ticks_in_deg(axi, 0.2, 0.2)
        axi.xaxis.set_ticks_position("both")
        axi.yaxis.set_ticks_position("both")
        axi.tick_params(axis="both", direction="in", length=5)

        annotate_corner(axi, chr(97 + i))
        i += 1

    for i in np.arange(Nx):
        for j in np.arange(Ny):
            if (i != 0) | (j != 0):
                ax[j, i].set_yticklabels([])
                ax[j, i].set_xticklabels([])

    mw = px_range[2]
    ms = py_range[0]

    ldx = 0.02
    ldy = ph + 0.015 * lx / ly
    lh = 0.015 * lx / ly
    lw = 0.005
    ax0.text(
        mw + ldx,
        ms + lh + ldy,
        "1 SD Credible Ellipse",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax0.annotate(
        "",
        xy=(mw + ldx, ms + ldy),
        xytext=(mw + pw / (pscale_cred) + ldx, ms + ldy),
        arrowprops=dict(
            arrowstyle="|-|,widthA=0.35,widthB=0.35", linewidth=1, shrinkA=0, shrinkB=0
        ),
    )
    ax0.text(
        mw + pw / pscale_cred + lw + ldx,
        ms + ldy,
        "1 m$\,$s$^{-1}$",
        ha="left",
        va="center",
        fontsize=10,
    )

    mw = px_range[1]
    ms = py_range[0]

    ax0.text(mw + ldx, ms + lh + ldy, "Velocity", ha="left", va="center", fontsize=10)
    ax0.annotate(
        "",
        xy=(mw + ldx, ms + ldy),
        xytext=(mw + pw / pscale + ldx, ms + ldy),
        arrowprops=dict(
            arrowstyle="<|-,widthA=0.35,widthB=0.35",
            linewidth=1,
            facecolor="k",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax0.text(
        mw + pw / pscale + lw + ldx,
        ms + ldy,
        "1 m$\,$s$^{-1}$",
        ha="left",
        va="center",
        fontsize=10,
    )

    ax0.axis("off")

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_noise_metrics(
    df_noise_gp_obs_t,
    df_noise_gp_num_t,
    df_noise_gos_t,
    plt_show=False,
    return_fig=False,
):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)

    ax[0].plot(
        df_noise_gos_t.noise_sd, df_noise_gos_t.RMSE, "rs", ms=4, lw=1, label="GOS"
    )
    ax[0].plot(
        df_noise_gp_num_t.noise_sd,
        df_noise_gp_num_t.RMSE,
        "ko",
        ms=4,
        lw=1,
        label="GP $\\tilde{\\theta}_{t}$",
    )
    ax[0].plot(
        df_noise_gp_obs_t.noise_sd,
        df_noise_gp_obs_t.RMSE,
        "bo",
        ms=4,
        lw=1,
        label="GP $\\hat{\\theta}_t$",
        mfc="w",
    )
    ax[0].set(xlabel="$\\sigma_\\tau$ (K)", ylabel="RMSE (m$\,$s$^{-1}$)")
    ylims = ax[0].get_ylim()
    ax[0].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)
    ax[0].legend(ncol=2, loc="upper center")

    ax[1].plot(
        df_noise_gp_num_t.noise_sd,
        df_noise_gp_num_t.coverage90,
        "ko",
        ms=4,
        lw=1,
        label="GP $\\tilde{\\theta}_{t}$",
    )
    ax[1].plot(
        df_noise_gp_obs_t.noise_sd,
        df_noise_gp_obs_t.coverage90,
        "bo",
        ms=4,
        lw=1,
        mfc="w",
        label="GP $\\hat{\\theta}_t$",
    )
    ax[1].set(xlabel="$\\sigma_\\tau$ (K)", ylabel="P90")
    ax[1].axhline(0.9, ls=":", lw=1, color="k")
    ylims = ax[1].get_ylim()
    ax[1].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)

    for i, axi in enumerate(ax.flatten()):
        axi.xaxis.set_ticks_position("both")
        axi.yaxis.set_ticks_position("both")
        axi.tick_params(axis="both", direction="in", length=5)
        axi.legend(ncol=2, loc="upper center")

        annotate_corner(axi, f"{chr(97 + i)})", box=False, buffer=4)

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_time_metrics(
    df_time_1h_gp_obs_t,
    df_time_1h_gp_num_t,
    df_time_1h_gp_num_t1,
    df_time_1h_gos_t,
    df_time_24h_gp_obs_t,
    df_time_24h_gp_num_t,
    df_time_24h_gp_num_t1,
    df_time_24h_gos_t,
    plt_show=False,
    return_fig=False,
):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

    dict_time_1h = {
        "gprm": df_time_1h_gp_num_t,
        "gprm_e": df_time_1h_gp_num_t1,
        "gos": df_time_1h_gos_t,
        "optimum": df_time_1h_gp_obs_t,
    }
    dict_time_24h = {
        "gprm": df_time_24h_gp_num_t,
        "gprm_e": df_time_24h_gp_num_t1,
        "gos": df_time_24h_gos_t,
        "optimum": df_time_24h_gp_obs_t,
    }

    type_test_list = ["time_1h", "time_24h"]
    row = 0
    for type_test in type_test_list:
        # make figure
        if type_test == "time_1h":
            xlabel = "$t$ (h)"
            xscale = 1 / 3600
            dict_time = dict_time_1h
        elif type_test == "time_24h":
            xlabel = "$t$ (d)"
            xscale = 1 / 3600 / 24
            dict_time = dict_time_24h

        ax[row, 0].plot(
            dict_time["gos"].time_sec * xscale,
            dict_time["gos"].RMSE,
            "r-",
            ms=4,
            lw=1,
            label="GOS",
        )
        ax[row, 0].plot(
            dict_time["gprm"].time_sec * xscale,
            dict_time["gprm"].RMSE,
            "k-",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_{t_1}$",
        )
        ax[row, 0].plot(
            dict_time["gprm_e"].time_sec * xscale,
            dict_time["gprm_e"].RMSE,
            "b-",
            ms=4,
            lw=1,
            label="GP $\\hat{\\theta}_t" + "$",
            mfc="w",
        )
        ax[row, 0].plot(
            dict_time["optimum"].time_sec * xscale,
            dict_time["optimum"].RMSE,
            "g-",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_t$",
        )
        ax[row, 0].set(xlabel=xlabel, ylabel="RMSE (m$\,$s$^{-1}$)")
        ylims = ax[row, 0].get_ylim()
        ax[row, 0].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)
        ax[row, 0].legend(ncol=2, loc="upper center")

        ax[row, 1].plot(
            dict_time["gprm"].time_sec * xscale,
            dict_time["gprm"].coverage90,
            "k-",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_{t_1}$",
        )
        ax[row, 1].plot(
            dict_time["gprm_e"].time_sec * xscale,
            dict_time["gprm_e"].coverage90,
            "b-",
            ms=4,
            lw=1,
            mfc="w",
            label="GP $\\hat{\\theta}_t" + "$",
        )
        ax[row, 1].plot(
            dict_time["optimum"].time_sec * xscale,
            dict_time["optimum"].coverage90,
            "g-",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_t$",
        )
        ax[row, 1].set(xlabel=xlabel, ylabel="P90")
        ax[row, 1].axhline(0.9, ls=":", lw=1, color="k")
        ylims = ax[row, 1].get_ylim()
        ax[row, 1].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)
        row += 1

    if len(type_test_list) == 2:
        for col in [0, 1]:
            ylim00 = ax[0, col].get_ylim()
            ylim10 = ax[1, col].get_ylim()
            ylims = (min(ylim00[0], ylim10[0]), max(ylim00[1], ylim10[1]))
            ax[0, col].set_ylim(ylims)
            ax[1, col].set_ylim(ylims)

    for i, axi in enumerate(ax.flatten()):
        axi.xaxis.set_ticks_position("both")
        axi.yaxis.set_ticks_position("both")
        axi.tick_params(axis="both", direction="in", length=5)
        axi.legend(ncol=2, loc="upper center")

        annotate_corner(axi, f"{chr(97 + i)})", box=False, buffer=4)

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_cloud_metrics(
    df_cloud_dense_gp_num_t,
    df_cloud_dense_gp_obs_t,
    df_cloud_sparse_gp_num_t,
    df_cloud_sparse_gp_obs_t,
    plt_show=False,
    return_fig=False,
):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

    dict_cloud_sparse = {
        "gprm": df_cloud_sparse_gp_num_t,
        "gprm_e": df_cloud_sparse_gp_obs_t,
    }
    dict_cloud_dense = {
        "gprm": df_cloud_dense_gp_num_t,
        "gprm_e": df_cloud_dense_gp_obs_t,
    }

    type_test_list = ["cloud_dense", "cloud_sparse"]
    row = 0
    for type_test in type_test_list:
        if type_test == "cloud_sparse":
            xlabel = "sparse cloud coverage ($\%$)"
            dict_cloud = dict_cloud_sparse
            val_name = "coverage_sparse"
        elif type_test == "cloud_dense":
            xlabel = "dense cloud coverage ($\%$)"
            dict_cloud = dict_cloud_dense
            val_name = "coverage_dense"

        ax[row, 0].plot(
            dict_cloud["gprm"][val_name] * 100,
            dict_cloud["gprm"].RMSE,
            "ko",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_{t_1}$",
        )
        ax[row, 0].plot(
            dict_cloud["gprm_e"][val_name] * 100,
            dict_cloud["gprm_e"].RMSE,
            "bo",
            ms=4,
            lw=1,
            label="GP $\\hat{\\theta}_t" + "$",
            mfc="w",
        )
        ax[row, 0].set(xlabel=xlabel, ylabel="RMSE (m$\,$s$^{-1}$)")
        ylims = ax[row, 0].get_ylim()
        ax[row, 0].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)
        ax[row, 0].legend(ncol=2, loc="upper center")

        ax[row, 1].plot(
            dict_cloud["gprm"][val_name] * 100,
            dict_cloud["gprm"].coverage90,
            "ko",
            ms=4,
            lw=1,
            label="GP $\\tilde{\\theta}_{t_1}$",
        )
        ax[row, 1].plot(
            dict_cloud["gprm_e"][val_name] * 100,
            dict_cloud["gprm_e"].coverage90,
            "bo",
            ms=4,
            lw=1,
            mfc="w",
            label="GP $\\hat{\\theta}_t" + "$",
        )
        ax[row, 1].set(xlabel=xlabel, ylabel="P90")
        ax[row, 1].axhline(0.9, ls=":", lw=1, color="k")
        ylims = ax[row, 1].get_ylim()
        ax[row, 1].set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.25)
        row += 1

    if len(type_test_list) == 2:
        for col in [0, 1]:
            ylim00 = ax[0, col].get_ylim()
            ylim10 = ax[1, col].get_ylim()
            ylims = (min(ylim00[0], ylim10[0]), max(ylim00[1], ylim10[1]))
            ax[0, col].set_ylim(ylims)
            ax[1, col].set_ylim(ylims)

    for i, axi in enumerate(ax.flatten()):
        axi.xaxis.set_ticks_position("both")
        axi.yaxis.set_ticks_position("both")
        axi.tick_params(axis="both", direction="in", length=5)
        axi.legend(ncol=2, loc="upper center")

        annotate_corner(axi, f"{chr(97 + i)})", box=False, buffer=4)

    if plt_show:
        plt.show()

    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def plot_psd(
    ax,
    psd_dict,
    kmin,
    kmax,
):
    mask = None

    # Plot multi fields
    for key, label, color in zip(
        ["model", "gp_mean"],
        ["OSSE", "GP mean"],
        ["red", "black"],
    ):
        k, psd = psd_dict[key]
        mask = (k >= kmin) & (k <= kmax)
        ax.plot(k[mask], psd[mask], label=label, color=color, lw=1)

    # Plot sample statistics
    k, mean = psd_dict["samples_mean"]
    _, sd = psd_dict["samples_sd"]

    mask = (k >= kmin) & (k <= kmax)

    ax.plot(
        k[mask],
        mean[mask],
        color="blue",
        label="GP samples",
        lw=1,
    )
    ax.fill_between(
        k[mask],
        mean[mask] - sd[mask],
        mean[mask] + sd[mask],
        color=[0.4, 0.5, 1],
        label=r"$\pm 1$ SD",
        zorder=0,
    )
    ax.fill_between(
        k[mask],
        mean[mask] - 2 * sd[mask],
        mean[mask] + 2 * sd[mask],
        color=[0.7, 0.8, 1],
        label=r"$\pm 2$ SD",
        zorder=-1,
    )
    # Plot theoretical PSD (if present)
    k_th, psd_th = psd_dict["theory"]
    ax.plot(
        k_th,
        psd_th,
        color="purple",
        lw=1,
        label="GP prior",
    )

    # Plot multi fields
    for key, label, color in zip(
        ["gos"],
        ["GOS"],
        ["green"],
    ):
        k, psd = psd_dict[key]
        mask = (k >= kmin) & (k <= kmax)
        ax.plot(k[mask], psd[mask], label=label, color=color, lw=1)

    ax.set_xlim(kmin, kmax)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("PSD")
    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax


def sci_factor_to_ylabel(ax):
    # Force draw so offset text is populated
    ax.figure.canvas.draw()
    offset_text = ax.yaxis.get_offset_text()
    raw = offset_text.get_text()

    if not raw:
        return

    # Extract exponent safely
    match = re.search(r"10\^\{([−-]?\d+)\}", raw)
    if not match:
        return

    exp = match.group(1).replace("−", "-")  # normalize minus
    ylabel = ax.get_ylabel().rstrip()

    # Append factor safely (respect existing math)
    new_ylabel = rf"{ylabel} $\times 10^{{{exp}}}$"
    ax.set_ylabel(new_ylabel)

    # Disable offset text
    offset_text.set_visible(False)


def plot_time_series_point(ds_point):
    # Extract variables
    To = ds_point.To.values
    dTds1o = ds_point.dTds1o.values
    dTds2o = ds_point.dTds2o.values
    dTdto = ds_point.dTdto.values
    muSstar = ds_point.muSstar.values
    stdSstar = ds_point.stdSstar.values
    muustar = ds_point.muustar.values
    stdustar = np.sqrt(ds_point.Kxstar_vel.values[:, 0, 0])
    muvstar = ds_point.muvstar.values
    stdvstar = np.sqrt(ds_point.Kxstar_vel.values[:, 1, 1])

    u = ds_point.u.values
    v = ds_point.v.values
    S = ds_point.S.values

    # Plotting parameters
    mn = 0.01
    mw = 0.08
    me = 0.01
    ms = 0.06

    Nx = 1
    Ny = 5

    aspect = 4
    lx = me + mw + Nx
    ly = mn + ms + Ny / aspect

    paspect = lx / ly

    ph = 1 / aspect / ly
    pw = 1 / lx

    px_range = (mw + np.arange(Nx)) / lx
    py_range = (ly - mn - 1 / aspect - np.arange(Ny) / aspect) / ly

    fw = 8
    fh = fw / paspect

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    time_date = ds_point.time.values
    time = (time_date - time_date[0]).astype("timedelta64[h]").astype(float)
    time_lims = (time.min(), time.max())

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes((px_range[i], py_range[jj], pw, ph))
            ax[jj, i].set_xlim(time_lims)

    # Plotting
    ax[0, 0].plot(time, To, "r-", lw=1, label=r"True $\mathbf{T}_t$")
    ax[0, 0].set_ylabel(r"$T_t$ ($^\circ$C)")
    ax[0, 0].legend(loc="lower right", fontsize=8)

    ax[1, 0].plot(
        time, dTds1o, "r-.", lw=1, label=r"True $\partial \mathbf{T}_t/\partial s_1$"
    )
    ax[1, 0].plot(
        time, dTds2o, "r--", lw=1, label=r"True $\partial \mathbf{T}_t/\partial s_2$"
    )
    ax[1, 0].set_ylabel(r"$\nabla T_t$ (K$\,$m$^{-1}$)")
    ax[1, 0].legend(loc="lower right", fontsize=8)

    ax[2, 0].plot(
        time, dTdto, "k:", lw=1.5, label=r"True $\partial \mathbf{T}_t/\partial t$"
    )
    ax[2, 0].plot(time, S, "r-", lw=1, label=r"True $\mathbf{S}_t$")
    ax[2, 0].plot(time, muSstar, "k-", lw=1, label=r"GP $\boldsymbol{\mu}_{S,t}^*$")
    ax[2, 0].fill_between(
        time, muSstar - stdSstar, muSstar + stdSstar, color=[0.6] * 3, label=r"GP 1 SD"
    )
    ax[2, 0].fill_between(
        time,
        muSstar - 2 * stdSstar,
        muSstar + 2 * stdSstar,
        color=[0.8] * 3,
        label=r"GP 2 SD",
        zorder=0,
    )
    ax[2, 0].set_ylabel(r"$S_t$ (K$\,$s$^{-1}$)")
    ax[2, 0].legend(loc="best", fontsize=8)

    ax[3, 0].plot(time, u, "r-", lw=1, label=r"True $\mathbf{u}_t$")
    ax[3, 0].plot(time, muustar, "k-", lw=1, label=r"GP $\boldsymbol{\mu}_{u,t}^*$")
    ax[3, 0].fill_between(
        time, muustar - stdustar, muustar + stdustar, color=[0.6] * 3, label=r"GP 1 SD"
    )
    ax[3, 0].fill_between(
        time,
        muustar - 2 * stdustar,
        muustar + 2 * stdustar,
        color=[0.8] * 3,
        label=r"GP 2 SD",
        zorder=0,
    )
    ax[3, 0].set_ylabel(r"$u_t$ (m$\,$s$^{-1}$)")
    ax[3, 0].legend(loc="upper right", fontsize=8, ncol=2)

    ax[4, 0].plot(time, v, "r-", lw=1, label=r"True $\mathbf{v}_t$")
    ax[4, 0].plot(time, muvstar, "k-", lw=1, label=r"GP $\boldsymbol{\mu}_{v,t}^*$")
    ax[4, 0].fill_between(
        time, muvstar - stdvstar, muvstar + stdvstar, color=[0.6] * 3, label=r"GP 1 SD"
    )
    ax[4, 0].fill_between(
        time,
        muvstar - 2 * stdvstar,
        muvstar + 2 * stdvstar,
        color=[0.8] * 3,
        label=r"GP 2 SD",
        zorder=0,
    )
    ax[4, 0].set_ylabel(r"$v_t$ (m$\,$s$^{-1}$)")
    ax[4, 0].legend(loc="upper right", fontsize=8, ncol=2)

    # Set u and v same ylim width
    ylims_u = ax[3, 0].get_ylim()
    width_u = ylims_u[1] - ylims_u[0]
    ylims_v = ax[4, 0].get_ylim()
    width_v = ylims_v[1] - ylims_v[0]
    max_width = max(width_u, width_v)
    center_u = (ylims_u[0] + ylims_u[1]) / 2
    center_v = (ylims_v[0] + ylims_v[1]) / 2
    ax[3, 0].set_ylim(center_u - max_width / 2, center_u + max_width / 2)
    ax[4, 0].set_ylim(center_v - max_width / 2, center_v + max_width / 2)

    # Labeling and aesthetics
    ax[Ny - 1, 0].set_xlabel("$t$ (h)")
    for i in range(Nx):
        for jj in range(Ny):
            annotate_corner(ax[jj, i], chr(97 + jj))

            ax[jj, i].xaxis.set_ticks_position("both")
            ax[jj, i].yaxis.set_ticks_position("both")
            ax[jj, i].tick_params(axis="both", direction="in", length=5)

            sci_factor_to_ylabel(ax[jj, i])

            if jj != Ny - 1:
                ax[jj, i].set_xticklabels([])

    ax0.axis("off")

    return fig, ax


def plot_spatial_time_series(
    LON,
    LAT,
    ug_mean,
    vg_mean,
    u_sample_mean,
    v_sample_mean,
    u_sample_std,
    v_sample_std,
    u_std,
    v_std,
    lonlims,
    latlims,
    plimspeed=None,
    plimvar=None,
):
    u_sample_var = u_sample_std**2
    v_sample_var = v_sample_std**2

    u_var = u_std**2
    v_var = v_std**2

    speed_mean = np.sqrt(ug_mean**2 + vg_mean**2)
    speed_star_mean = np.sqrt(u_sample_mean**2 + v_sample_mean**2)

    pscale = 2.5
    nx = ny = 16

    mn = 0.1
    mw = 0.2
    me = 0.22
    ms = 0.3

    td = 0.005

    Nx = 3
    Ny = 2

    aspect = 1
    lx = me + mw + Nx
    ly = mn + ms + Ny / aspect

    paspect = lx / ly

    ph = 1 / aspect / ly
    pw = 1 / lx

    cbd = 0.01 * lx / ly
    cbr = 0.85
    cbw = 0.015 * lx / ly

    px_range = (mw + np.arange(Nx)) / lx
    py_range = (ly - mn - 1 / aspect - np.arange(Ny) / aspect) / ly

    fw = 8
    fh = fw / paspect

    ax = np.empty([Ny, Nx], object)

    fig = plt.figure(figsize=(fw, fh))

    ax0 = fig.add_axes((0, 0, 1, 1))
    ax0.set_xlim((0, 1))
    ax0.set_ylim((0, 1))

    for i in range(Nx):
        for jj in range(Ny):
            ax[jj, i] = fig.add_axes((px_range[i], py_range[jj], pw, ph))
            ax[jj, i].set_xlim(lonlims)
            ax[jj, i].set_ylim(latlims)

    if plimspeed is None:
        listvar = [
            np.sqrt(ug_mean**2 + vg_mean**2).flatten(),
            np.sqrt(u_sample_mean**2 + v_sample_mean**2).flatten(),
        ]
        plimvalspeed = np.hstack(listvar)
        plimspeed = [0, np.nanmax(plimvalspeed)]

    if plimvar is None:
        listvar = [
            u_var.flatten(),
            v_var.flatten(),
            u_sample_var.flatten(),
            v_sample_var.flatten(),
        ]
        plimvalvar = np.hstack(listvar)
        plimvar = [0, np.nanmax(plimvalvar)]

    cmapvar = cmocean.cm.matter
    cmapspeed = cmocean.cm.speed

    add_colorbar(
        fig,
        ax[1, 0],
        "bottom",
        plimspeed,
        cmapspeed,
        cbr,
        cbd,
        cbw,
        label=r"(m$\,$s$^{-1}$)",
    )
    add_colorbar(
        fig,
        ax[1, 1],
        "bottom",
        plimvar,
        cmapvar,
        cbr,
        cbd,
        cbw,
        label=r"(m$^2\,$s$^{-2}$)",
        extend="max",
    )

    imshow(
        ax[0, 0],
        LON,
        LAT,
        speed_mean,
        cmap=cmapspeed,
        vmin=plimspeed[0],
        vmax=plimspeed[1],
    )
    quiver(
        ax[0, 0],
        LON,
        LAT,
        ug_mean,
        vg_mean,
        scale=pscale,
        unit=0.2,
        xlims=lonlims,
        ylims=latlims,
        nx=nx,
        ny=ny,
        an=False,
        label=r"m\:s$^{-1}$",
    )

    imshow(ax[0, 1], LON, LAT, u_var, cmap=cmapvar, vmin=plimvar[0], vmax=plimvar[1])

    imshow(ax[0, 2], LON, LAT, v_var, cmap=cmapvar, vmin=plimvar[0], vmax=plimvar[1])

    imshow(
        ax[1, 0],
        LON,
        LAT,
        speed_star_mean,
        cmap=cmapspeed,
        vmin=plimspeed[0],
        vmax=plimspeed[1],
    )
    quiver(
        ax[1, 0],
        LON,
        LAT,
        u_sample_mean,
        v_sample_mean,
        scale=pscale,
        unit=0.2,
        xlims=lonlims,
        ylims=latlims,
        nx=nx,
        ny=ny,
        an=False,
    )

    imshow(
        ax[1, 1], LON, LAT, u_sample_var, cmap=cmapvar, vmin=plimvar[0], vmax=plimvar[1]
    )
    imshow(
        ax[1, 2], LON, LAT, v_sample_var, cmap=cmapvar, vmin=plimvar[0], vmax=plimvar[1]
    )

    i = 0
    for axi in ax.flatten():
        if axi.has_data():
            ticks_in_deg(axi, 1, 1)
            axi.xaxis.set_ticks_position("both")
            axi.yaxis.set_ticks_position("both")
            axi.tick_params(axis="both", direction="in", length=5)
            axi.set_facecolor("white")

            annotate_corner(axi, chr(97 + i))
            i += 1

    jticks = 1
    iticks = 2
    for i in range(Nx):
        for j in range(Ny):
            if (i != iticks) or (j != jticks):
                ax[j, i].set_xticklabels([])
                ax[j, i].set_yticklabels([])
            else:
                ax[j, i].yaxis.set_label_position("right")
                ax[j, i].yaxis.tick_right()

    ax0.text(
        px_range[0] - td,
        py_range[0] + ph / 2,
        "Observing system\nsimulation experiment data",
        ha="right",
        va="center",
        fontsize=12,
        rotation=90,
        multialignment="center",
    )
    ax0.text(
        px_range[0] - td,
        py_range[1] + ph / 2,
        "GP prediction",
        ha="right",
        va="center",
        fontsize=12,
        rotation=90,
    )
    ax0.text(
        px_range[0] + pw / 2,
        py_range[0] + td + ph,
        "Mean velocity",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    ax0.text(
        px_range[1] + pw / 2,
        py_range[0] + td + ph,
        r"Variance of $u_t(\cdot)$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    ax0.text(
        px_range[2] + pw / 2,
        py_range[0] + td + ph,
        r"Variance of $v_t(\cdot)$",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    mw = px_range[0]
    ms = py_range[1]

    ldy = -0.08 * lx / ly
    lh = 0.015 * lx / ly

    ax0.annotate(
        "",
        xy=(mw, ms + ldy),
        xytext=(mw + pw / pscale * 0.5, ms + ldy),
        arrowprops=dict(
            arrowstyle="<|-,widthA=0.35,widthB=0.35",
            linewidth=1,
            facecolor="k",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax0.text(
        mw, ms + lh + ldy, r"0.5 m$\,$s$^{-1}$", ha="left", va="center", fontsize=10
    )

    ax0.axis("off")
    return fig, ax


def plot_single_psd(ax, psd_dict, key, label, color, kmin, kmax):
    k, psd = psd_dict[key]
    mask = (k >= kmin) & (k <= kmax)
    ax.plot(k[mask], psd[mask], label=label, color=color, lw=1)
    pass


def plot_psd_overview(ax, psd_dict, kmin, kmax, parameter):
    plot_single_psd(ax, psd_dict, "model", "Truth", "red", kmin, kmax)

    # Plot sample statistics
    def plot_samples_psd(ax, psd_dict, kmin, kmax):
        k, mean = psd_dict["samples_mean"]
        _, sd_1lower = psd_dict["samples_1sigma_lower"]
        _, sd_1upper = psd_dict["samples_1sigma_upper"]
        _, sd_2lower = psd_dict["samples_2sigma_lower"]
        _, sd_2upper = psd_dict["samples_2sigma_upper"]

        mask = (k >= kmin) & (k <= kmax)

        ax.plot(
            k[mask],
            mean[mask],
            color="k",
            label="GP samples",
            lw=1,
        )
        ax.fill_between(
            k[mask],
            sd_1lower[mask],
            sd_1upper[mask],
            color=[0.65, 0.65, 0.65],
            label=r"$\pm 1$ SD",
            zorder=0,
        )
        ax.fill_between(
            k[mask],
            sd_2lower[mask],
            sd_2upper[mask],
            color=[0.8, 0.8, 0.8],
            label=r"$\pm 2$ SD",
            zorder=-1,
        )

    plot_samples_psd(ax, psd_dict, kmin, kmax)

    k_theory, psd_theory = psd_dict["theory"]
    ax.plot(k_theory, psd_theory, "-", color="blue", lw=1, label=r"GP cov fun")

    k_theory_no_noise, psd_theory_no_noise = psd_dict["theory_no_noise"]
    ax.plot(
        k_theory_no_noise,
        psd_theory_no_noise,
        "--",
        color="blue",
        lw=1,
        label=rf"GP cov fun ($\gamma^2_{{{parameter},t}}=0$)",
    )[0].set_dashes([5, 5])

    plot_single_psd(ax, psd_dict, "gos", "GOS", "green", kmin, kmax)

    ax.set_xlim(kmin, kmax)
    ax.set_xlabel(r"$k$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$k$ (m$^{-1}$)")

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="both", which="major", direction="in", length=5)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)
    pass
