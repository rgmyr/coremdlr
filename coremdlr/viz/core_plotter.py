"""
Core data plotting API.

Copied this file out here because I plan to make it into its own package.
"""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from coremdlr.config import strip_config


# tuple subclass for individual axis
# TODO: integrate this instead of using lists
#AxisSpec = namedtuple('AxisSpec', ['name', 'feature', 'geometry'])


class CorePlotter():
    """
    Class to handle figure and axes creation, ticks, saving, and plotting common core datatypes.
    """
    def __init__(self, figsize=(15,650)):

        self.fig = plt.figure(figsize=figsize)
        self.ax = []
        self.ax_names = []
        self.ncols = 0


    def plot_image(self, image, depths=None, **kwargs):
        """
        Plot an image.
        """
        ax = self._add_new_ax(name=kwargs.pop('name', 'Image'))

        if depths is not None:
            self._set_ylim(depths[0], depths[-1])

            resolution = kwargs.pop('resolution', image.shape[0] // depths.size)

            tick_lists = self._make_image_ticks(depths, resolution=resolution, **kwargs)
            major_ticks, major_locs, minor_ticks, minor_locs = tick_lists

            ax.yaxis.set_major_formatter(ticker.FixedFormatter((major_ticks)))
            ax.yaxis.set_major_locator(ticker.FixedLocator((major_locs)))

            ax.yaxis.set_minor_formatter(ticker.FixedFormatter((minor_ticks)))
            ax.yaxis.set_minor_locator(ticker.FixedLocator((minor_locs)))

            ax.tick_params(which='major', labelsize=kwargs.get('major_tick_size', 16), color='black')
            ax.tick_params(which='minor', labelsize=kwargs.get('minor_tick_size',  8), color='gray')

        ax.grid(False)
        ax.imshow(image)

        return self.fig, self.ax


    def plot_log(self, depths, log, existing_ax=False, scatter=False, **kwargs):
        """
        Plot a log or log-like data.

        Parameters
        ----------
        depths : array
            Array of depths, shape=(n_samples,)
        log : array
            Array of log values, shape=(n_samples,)
        name : str, optional
            Name to use for axis, or to reference existing axis.
        scatter : bool, optional
            If True, plot as scatter, otherwise plot a line. Default=False.
        **kwargs : optional
            Args for plot/scatter/tick functions. Use `fmt` arg for format strings.
        """
        #ax = self.get_ax_by_name(kwargs.pop('name', 'Logs'))
        #ax = self._add_new_ax(name=kwargs.pop('name', 'Logs'))
        ax = None
        # TODO: fix this part

        top = getattr(self, 'top', depths[0])
        base = getattr(self, 'base', depths[-1])

        if not existing_ax or (ax is None):
            ax = self._add_new_ax(name='Logs')
            ax.invert_yaxis()

            precision = kwargs.get('precision', 0.1)
            ax.yaxis.set_ticks(np.arange(base, top-precision, -precision))

        else:
            ax = ax.twiny()

        lmin, lmax = log.min(), log.max()
        dx = lmax - lmin
        ax.set_xlim(lmin-0.1*dx, lmax+0.1*dx)
        ax.set_ylim(base, top)

        ax.tick_params(axis='x', labelcolor=kwargs.get('color', 'blue'))

        ax.grid(True)

        if scatter:
            ax.scatter(log, depths, kwargs.pop('fmt', '.'), **kwargs)
        else:
            ax.plot(log, depths, kwargs.pop('fmt', '-'), **kwargs)

        return self.fig, self.ax


    def plot_strip(self, striplog, name=None, legend=strip_config.legend):
        """
        Plot a striplog on a new axis.

        Parameters
        ----------
        striplog : strilog.StripLog instance
            The labels to plot as a Striplog.
        name : str, optional
            Name for axis.
        legend : striplog.Legend, optional
            Legend to use for striplog, if not the default.
        """
        ax = self._add_new_ax(name=name)
        striplog[0].top = self.top
        striplog[-1].base = self.base
        striplog.plot(ax=ax, legend=legend)

        return self.fig, self.ax


    def get_ax_by_name(self, name):
        """
        Return a named Axis instance. If not found, return None.
        """
        try:
            return self.ax[self.ax_names.index(name)]
        except ValueError:
            print(f'No axis with name: {name}. Returning None.')
            return None


    def _set_ylim(self, top, base):

        self.top = top
        self.base = base


    def _add_new_ax(self, name=None):
        """
        Add an axis to self.axes and return it. For now, always added to the right.
        """
        self.ncols += 1
        for i, (ax, ax_name) in enumerate(zip(self.fig.axes, self.ax_names)):
            #print(f'Changing ax {ax_name} geom to: {(1,self.ncols,i+1)}')
            ax.change_geometry(1, self.ncols, i+1)

        new_ax = self.fig.add_subplot(1, self.ncols, self.ncols)
        self.ax.append(new_ax)
        self.ax_names.append(name)

        return new_ax

    def _make_image_ticks(self, depths,
                          resolution=1, top=None, base=None,
                          major_precision=0.1,
                          major_format_str='{:.1f}',
                          minor_precision=0.01,
                          minor_format_str='{:.2f}'):
        """
        Generate major & minor (ticks, locs) for an image axis.

        Parameters
        ----------
        depths: array
            1D array of depths
        resolution : int, optional
            Number of pixel rows per (equally spaced) value in `depths`, default=1.
        top : float, optional
            Depth tick value to place at top of image, default=None.
        base: float, optional
            Depth tick value to place at bottom of image, default=None.
        precision : float, optional
            Tick spacing (in depth units), default=0.01.
        format_str : str, optional
            Format string to coerce float depths -> tick strings, default='{:.2f}'.

        Returns
        -------
        major_ticks, major_locs, minor_ticks, minor_locs

        ticks : list of tick strings
        locs : list of tick locations in image data coordinates (fractional row indices)
        """
        # lambdas to convert values --> strs
        major_fmt_fn = lambda x: major_format_str.format(x)
        minor_fmt_fn = lambda x: minor_format_str.format(x)

        major_ticks, major_locs = [], []
        minor_ticks, minor_locs = [], []

        # remainders w.r.t. precision
        major_rmndr = np.insert(depths % major_precision, (0, depths.size), np.inf)
        minor_rmndr = np.insert(depths % minor_precision, (0, depths.size), np.inf)

        for i in np.arange(1, depths.size+1):

            if np.argmin(major_rmndr[i-1:i+2]) == 1:
                major_ticks.append(major_fmt_fn(depths[i-1]))
                major_locs.append(i*resolution + resolution // 2)

            elif np.argmin(minor_rmndr[i-1:i+2]) == 1:
                if major_ticks[-1]+'0' == minor_fmt_fn(depths[i-1]):
                    # fixes some overlapping ticks, BUT not robust
                    # enough for non-default precision combos
                    continue
                minor_ticks.append(minor_fmt_fn(depths[i-1]))
                minor_locs.append(i*resolution + resolution // 2)

        return major_ticks, major_locs, minor_ticks, minor_locs
